import yaml
from typing import NamedTuple, Any
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.profiler import TraceAnnotation
from jax.experimental.host_callback import id_print
import haiku as hk
import optax
from data_loading import get_loader 
from generate_data import generate_image 
from functools import partial

import wandb
from tqdm import tqdm

import sys
import augmax

import models
import loss_functions
from plotting import log_image, log_anim


METRICS = dict(
    l1 = loss_functions.l1_loss,
    l2 = loss_functions.l2_loss,
    min_min = loss_functions.min_min_loss,
    dtw = loss_functions.DTW(),
    forward_mae = loss_functions.forward_mae,
    backward_mae = loss_functions.backward_mae,
    forward_rmse = loss_functions.forward_rmse,
    backward_rmse = loss_functions.backward_rmse,
)
PATIENCE = 100


def get_optimizer():
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=1e-3,
        warmup_steps=1024,
        decay_steps=40000-1024,
        end_value=1e-4
    )
    return optax.adam(lr_schedule, b1=0.5, b2=0.9)


class TrainingState(NamedTuple):
    params: hk.Params
    buffers: hk.State
    opt: optax.OptState


def changed_state(state, params=None, buffers=None, opt=None):
    return TrainingState(
        params = state.params if params is None else params,
        buffers = state.buffers if buffers is None else buffers,
        opt = state.opt if opt is None else opt,
    )


def prep(batch, key=None, augment=False, input_types=None):
    ops = []
    if augment: ops += [
        augmax.HorizontalFlip(),
        augmax.VerticalFlip(),
        augmax.Rotate90(),
        augmax.Rotate(15),
        # augmax.Warp(coarseness=16)
    ]
    ops += [augmax.ByteToFloat()]
    # if augment: ops += [
    #     augmax.ChannelShuffle(p=0.1),
    #     augmax.Solarization(p=0.1),
    # ]

    if input_types is None:
        input_types = [
            augmax.InputType.IMAGE,
            augmax.InputType.CONTOUR,
        ]
    chain = augmax.Chain(*ops, input_types=input_types)
    if augment == False:
        key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, batch[0].shape[0])
    transformation = jax.vmap(chain)
    outputs = list(transformation(subkeys, *batch))
    for i, typ in enumerate(input_types):
        if typ == augmax.InputType.CONTOUR:
            outputs[i] = 2 * (outputs[i] / outputs[0].shape[1]) - 1.0

    return outputs


def update_net(state, loss_closure):
    _, optimizer = get_optimizer()
    (loss, (buffers, output, metrics)), gradients = jax.value_and_grad(loss_closure, has_aux=True)(state.params)
    updates, opt = optimizer(gradients, state.opt, state.params)
    params = optax.apply_updates(state.params, updates)
    return TrainingState(params, buffers, opt), output, metrics


@partial(jax.jit, static_argnums=3)
def train_step(batch, state, key, net):
    _, optimizer = get_optimizer()

    aug_key, model_key = jax.random.split(key)
    img, snake = prep(batch, aug_key, augment=True)

    def calculate_loss(params):
        (snake_preds, aux), buffers = net(params, state.buffers, model_key, img, is_training=True)

        loss_terms = {}
        for i, pred in enumerate(snake_preds, 1):
            loss = jnp.mean(jax.vmap(loss_fn)(pred, snake))
            loss_terms[f'snake_loss_step{i}'] = loss

        return sum(loss_terms.values()), (buffers, snake_preds[-1], loss_terms)

    (loss, (buffers, prediction, metrics)), gradients = jax.value_and_grad(calculate_loss, has_aux=True)(state.params)
    updates, new_opt = optimizer(gradients, state.opt, state.params)
    new_params = optax.apply_updates(state.params, updates)

    metrics['loss'] = loss
    for m in METRICS:
        metrics[m] = jnp.mean(jax.vmap(METRICS[m])(prediction, snake))

    return metrics, changed_state(state,
        params=new_params,
        buffers=buffers,
        opt=new_opt,
    )


@partial(jax.jit, static_argnums=3)
def val_step(batch, state, key, net):
    imagery, contours = prep(batch)
    (predictions, _), _ = net(state.params, state.buffers, key, imagery, is_training=False)

    metrics = {}
    for i, pred in enumerate(predictions, 1):
        loss = jnp.mean(jax.vmap(loss_fn)(pred, contours))
        metrics[f'snake_loss_step{i}'] = loss

    pred = predictions[-1]
    for m in METRICS:
        metrics[m] = jnp.mean(jax.vmap(METRICS[m])(pred, contours))

    return metrics, imagery, contours, predictions


def save_state(state, out_path):
    state = jax.device_get(state)
    with out_path.open('wb') as f:
        pickle.dump(state, f)


def log_metrics(metrics, prefix, epoch):
    metrics = {m: np.mean(metrics[m]) for m in metrics}

    wandb.log({f'{prefix}/{m}': metrics[m] for m in metrics}, step=epoch)
    print(f'{prefix}/metrics')
    print(', '.join(f'{k}: {v:.3f}' for k, v in metrics.items()))


if __name__ == '__main__':
    train_key = jax.random.PRNGKey(42)
    persistent_val_key = jax.random.PRNGKey(27)
    multiplier = 64

    config = yaml.load(open('config.yml'), Loader=yaml.SafeLoader)
    lf = config['loss_function']
    if lf.startswith('SoftDTW'):
        gamma = float(lf[8:-1])
        loss_fn = loss_functions.SoftDTW(gamma=gamma)
    elif lf == 'DTW':
        loss_fn = loss_functions.DTW()
    else:
        loss_fn = getattr(loss_functions, lf)

    model_args = config['model_args']
    model_args['vertices'] = config['vertices']
    modelclass = getattr(models, config['model'])
    S = modelclass(**model_args)
    S = hk.transform_with_state(S)

    # initialize data loading
    train_key, subkey = jax.random.split(train_key)
    train_loader = get_loader(config['batch_size'], 4, 'train', subkey)
    val_loader   = get_loader(16, 1, 'validation', None)
    img, snake = prep(next(iter(train_loader)))

    # Initialize model and optimizer state
    opt_init, _ = get_optimizer()
    params, buffers = S.init(jax.random.PRNGKey(39), img[:1], is_training=True)
    state = TrainingState(params=params, buffers=buffers, opt=opt_init(params))
    net = S.apply

    running_min = np.inf
    last_improvement = 0
    wandb.init(project='DeepSnake CALFIN', config=config)

    for epoch in range(1, 1001):
        wandb.log({f'epoch': epoch}, step=epoch)
        prog = tqdm(train_loader, desc=f'Ep {epoch} Trn')
        trn_metrics = {}
        loss_ary = None
        for step, batch in enumerate(prog, 1):
            train_key, subkey = jax.random.split(train_key)
            metrics, state = train_step(batch, state, subkey, net)

            for m in metrics:
              if m not in trn_metrics: trn_metrics[m] = []
              trn_metrics[m].append(metrics[m])

        log_metrics(trn_metrics, 'trn', epoch)
        if epoch % 10 != 0:
            continue

        # Save Checkpoint
        ckpt_dir = Path('checkpoints')
        ckpt_dir.mkdir(exist_ok=True)
        save_state(state, ckpt_dir / f'{wandb.run.id}-latest.npz')
        save_state(state, ckpt_dir / f'{wandb.run.id}-{epoch:04d}.npz')

        # Validate
        val_key = persistent_val_key
        val_metrics = {}
        for step, batch in enumerate(val_loader):
            val_key, subkey = jax.random.split(val_key)
            metrics, *inspection = val_step(batch, state, subkey, net)
            for m in metrics:
              if m not in val_metrics: val_metrics[m] = []
              val_metrics[m].append(metrics[m])

            imagery, contours, predictions = inspection
            predictions = [p[0] for p in predictions]
            log_image(imagery[0], contours[0], predictions, f"Imgs/{step}", epoch)
            log_anim(imagery[0], contours[0], predictions, f"Animated/{step}", epoch)

        log_metrics(val_metrics, 'val', epoch)
