"""Training objective, update and noise: checked against numbers of the original training code.

The reference numbers below were produced by the code that trained the released students (its
update functions, LoRA module and data path), run on the tiny models and random data of
``train_tiny_models.py`` on a CPU: 3 updates of 4 accumulated micro-batches, rank-4 adapters,
AdamW lr 1e-2, clip 10, EMA 0.99. On the same machine the port reproduces them bit for bit (CPU and
GPU with bfloat16 autocast); here a small tolerance allows for other CPUs and math libraries.
"""
import math
import random

import pytest
import torch

from fastpixal3d.training import objective as O
from fastpixal3d.training import schedule as S
from fastpixal3d.training import student as ST
from train_tiny_models import Sparse, TinyDenseFlow, TinySparseFlow, structure_batches, texture_batches

STRUCTURE = {
    'losses': [1.6093941926956177, 1.639668345451355, 1.5963855981826782],
    'grad_norms': [0.033821962773799896, 0.06905677169561386, 0.030376968905329704],
    'param_sums': [-0.18968488555401564, 0.22001791163347661, 0.507997015491128, -0.18426719098351896, -1.0220102798193693, 0.18649082339834422, 1.968782494775951, 0.0758005992975086, 1.0016498817130923, -0.4916826474945992, -0.4640873554162681, 0.0751034040004015, -0.7042872458696365, -0.00886910269036889, -1.167824603850022, -0.16263593244366348],
    'ema_sums': [-0.18211079551838338, 0.004345200395619031, 0.5121631994843483, -0.0028066658196621574, -0.9424431538791396, 0.004624488414265215, 2.0609054071828723, 0.000483071940834634, 0.9923685160465539, -0.009626863480661996, -0.5775102553889155, 0.0009383401047671214, -0.6677493005990982, 4.0795050153974444e-05, -1.193695189198479, -0.0030672596185468137],
}
TEXTURE = {
    'losses': [2.205462634563446, 2.143933057785034, 2.134583592414856],
    'grad_norms': [0.040902115404605865, 0.03890344500541687, 0.03211624547839165],
    'param_sums': [-0.9812511992640793, -0.32617352437227964, 1.8454218599945307, -0.16433925204910338, -1.3903741561807692, 0.03356981498654932, -0.1526014287956059, -0.021246402990072966, 0.6504422361031175, -0.12499589659273624, 0.6394023825414479, 0.042480414267629385, -0.5579382148571312, 0.6358826889190823, 3.906972930417396, 0.05048414901830256],
    'ema_sums': [-1.2094910782761872, -0.004597008686687332, 1.8556963557493873, -0.0036471165367402136, -1.2999930352380034, 0.0012384413348627277, -0.6646023850116762, -0.0010303852905053645, 0.6774937680293078, -0.004039439452753868, 0.6222305174451321, 0.0005846130006830208, -0.4330269618294551, 0.012452378257876262, 3.140899846534012, 0.0027401569968787953],
}


def _close(got, want, rel=1e-5, abs_=1e-6):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert math.isclose(g, w, rel_tol=rel, abs_tol=abs_), (got, want)


def test_structure_updates_match_original_training_code():
    gen = torch.Generator().manual_seed(99)
    batches = structure_batches(gen)
    torch.manual_seed(1234)
    flow = TinyDenseFlow().requires_grad_(False).eval()
    net = ST.StructureStudentNetwork(flow, ST.attach_adapters(flow, 4, 4.0, 0.0))
    params = net.trainable()
    opt = O.make_optimizer(params, 1e-2, (0.9, 0.999), 0.0)
    ema = O.init_ema(params)
    losses, norms = [], []
    for batch in batches:
        mbs = [O.MicroBatch("v", 0, 0, ti, tj, x, t, {"cond": c}) for x, ti, tj, t, c in batch]
        r = O.run_update(lambda mb: net(mb.x_i, mb.t_i, mb.inputs["cond"]), mbs, params, opt, ema,
                         grad_clip=10.0, ema_decay=0.99, autocast_dtype=None, device_type="cpu")
        losses.append(r.loss)
        norms.append(r.grad_norm)
    _close(losses, STRUCTURE["losses"])
    _close(norms, STRUCTURE["grad_norms"])
    _close([float(p.double().sum()) for p in params.values()], STRUCTURE["param_sums"])
    _close([float(p.double().sum()) for p in ema.values()], STRUCTURE["ema_sums"])


def test_texture_updates_match_original_training_code():
    gen = torch.Generator().manual_seed(99)
    structure_batches(gen)  # the reference data drew the structure batches first
    batches = texture_batches(gen)
    torch.manual_seed(1235)
    flow = TinySparseFlow().requires_grad_(False).eval()
    net = ST.TextureStudentNetwork(flow, ST.attach_adapters(flow, 4, 4.0, 0.0))
    params = net.trainable()
    opt = O.make_optimizer(params, 1e-2, (0.9, 0.999), 0.0)
    ema = O.init_ema(params)

    def prep(mb):
        x = mb["x"]
        return O.MicroBatch("v", mb["i"], 0, mb["ti"], mb["tj"], x, mb["target"].float(),
                            {"x": Sparse(x, mb["coords"]), "cond": {"global": mb["global"]},
                             "concat": Sparse(mb["concat"], mb["coords"])})

    losses, norms = [], []
    for batch in batches:
        r = O.run_update(lambda mb: net(mb.inputs["x"], mb.t_i, mb.inputs["cond"], mb.inputs["concat"]).feats, batch,
                         params, opt, ema, prepare=prep, grad_clip=10.0, ema_decay=0.99, autocast_dtype=None,
                         device_type="cpu")
        losses.append(r.loss)
        norms.append(r.grad_norm)
    _close(losses, TEXTURE["losses"])
    _close(norms, TEXTURE["grad_norms"])
    _close([float(p.double().sum()) for p in params.values()], TEXTURE["param_sums"])
    _close([float(p.double().sum()) for p in ema.values()], TEXTURE["ema_sums"])


def test_loss_is_mse_after_one_euler_step():
    g = torch.Generator().manual_seed(0)
    x, v, tgt = (torch.randn(10, 3, generator=g) for _ in range(3))
    loss = O.trajectory_loss(x.bfloat16(), v.bfloat16(), 0.9, 0.75, tgt)
    want = ((x.bfloat16().float() - (0.9 - 0.75) * v.bfloat16().float() - tgt) ** 2).mean()
    assert loss.dtype == torch.float32 and torch.allclose(loss, want, rtol=1e-6, atol=0)


def test_gradient_accumulation_equals_mean_gradient():
    """4 micro-batches of loss/4 on one process = gradient of the mean loss (= 4 processes x 1, averaged)."""
    g = torch.Generator().manual_seed(1)
    xs = [torch.randn(5, 3, generator=g) for _ in range(4)]
    ts = [torch.randn(5, 3, generator=g) for _ in range(4)]
    w0 = torch.randn(3, 3, generator=g)

    def grads(accumulate):
        w = torch.nn.Parameter(w0.clone())
        if accumulate:
            for x, t in zip(xs, ts):
                (O.trajectory_loss(x, x @ w, 1.0, 0.75, t) / 4).backward()
        else:
            per_process = []
            for x, t in zip(xs, ts):
                wi = torch.nn.Parameter(w0.clone())
                O.trajectory_loss(x, x @ wi, 1.0, 0.75, t).backward()
                per_process.append(wi.grad)
            return sum(per_process) / 4
        return w.grad

    assert torch.allclose(grads(True), grads(False), atol=1e-7)


def test_draw_order_is_choice_then_randrange():
    views, pairs = [f"v{i}" for i in range(7)], [(0, 3), (3, 6), (6, 8), (8, 12)]
    rng, ref = random.Random(5), random.Random(5)
    got = O.draw_segments(rng, views, pairs, 6)
    want = []
    for _ in range(6):
        v = ref.choice(views)
        want.append((v, pairs[ref.randrange(len(pairs))]))
    assert got == want


def test_grids_of_the_released_students():
    assert S.grid_t_values([0, 3, 6, 8, 12]) == [1.0, 0.9, 0.75, 0.6, 0.0]
    assert S.grid_t_values([0, 6, 12]) == [1.0, 0.75, 0.0]
    assert S.segment_pairs([0, 3, 6, 8, 12]) == [(0, 3), (3, 6), (6, 8), (8, 12)]
    assert S.uniform_grid_indices(2) == [0, 6, 12]
    with pytest.raises(ValueError):
        S.check_grid_indices([0, 5, 12], stored=(2, 3, 4, 6, 8, 9, 10, 12))
    with pytest.raises(ValueError):
        S.check_grid_indices([0, 6, 6, 12])
    for k in (1, 2, 3, 4, 6, 12):  # k-step grids are exact subsequences of the 12-step grid
        assert (S.teacher_t_grid(k) == S.teacher_t_grid(12)[S.uniform_grid_indices(k)]).all()


def test_noise_seeds_of_released_anchors():
    # the seed recorded in a structure anchor record of the released run, for this view
    assert S.stable_seed(S.STRUCTURE_NOISE_SALT, "0ca0115101a241df8ef1f5d45047b926.view01") == 5528962649559725838
    a = S.structure_initial_noise("x.view00")
    assert a.shape == (1, 8, 16, 16, 16) and torch.equal(a, S.structure_initial_noise("x.view00"))
    torch.manual_seed(3)
    b = S.texture_initial_noise(11)
    torch.manual_seed(4)
    assert torch.equal(b, S.texture_initial_noise(11))  # independent of the global generator


def test_full_grid_order_and_gradients():
    g = torch.Generator().manual_seed(2)
    dense = torch.randn(1, 8, 16, 16, 16, generator=g, requires_grad=True)
    index = S.full_grid_index(16)
    tokens = S.full_grid_gather(dense, index)
    assert torch.equal(tokens, dense.detach().reshape(8, -1).t())  # row x*256 + y*16 + z
    assert torch.equal(S.full_grid_scatter(tokens.detach(), index), dense.detach())
    w = torch.randn(4096, 8, generator=g)
    (tokens * w).sum().backward()
    assert torch.equal(dense.grad, w.t().reshape(1, 8, 16, 16, 16))
