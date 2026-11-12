"""Consistency distillation of the shape students: checked against numbers of the original training code.

The reference numbers below were produced by the code that trained the released coarse- and fine-shape
students (its ``cd_init_loss`` and ``teacher_ode_x0``, guided teacher velocity, EMA with the decay ramp and
AdamW), run on the tiny shape flow of ``train_tiny_models.py`` with Pixal3D's sparse tensor on a CPU: three
updates of one object each, rank-4 adapters, AdamW lr 1e-2 (betas (0, 0.999), weight decay 1e-3), clip 10,
EMA 0.99 with the ramp, loss weight 80. On the same machine the port reproduces them bit for bit; here a
small tolerance allows for other CPUs and math libraries.
"""
import math

import pytest
import torch

from fastpixal3d.training import consistency as CD
from fastpixal3d.training import objective as O
from fastpixal3d.training import student as ST
from train_tiny_models import Sparse, TinyShapeFlow, shape_objects

RUN_SEED = 20260716
ORDER = ["obj01", "obj03", "obj00"]
REFERENCE = {
    'losses': [0.002423819579664395, 0.0019051281575573414, 0.00023964678749383733],
    'grad_norms': [0.01594354957342148, 0.021386565640568733, 0.0023922286927700043],
    't_n': [0.46000003814697266, 0.4399999976158142, 0.24000000953674316],
    'target_std': [0.6243705749511719, 0.6702972054481506, 0.7151284217834473],
    'student_std': [0.6243080496788025, 0.6702496409416199, 0.7145435810089111],
    'anchor_sums': {"obj01": -20.135823699645698, "obj03": 191.19776388653554, "obj00": 28.76575900055468},
    'param_sums': [0.6939268105197698, 0.044990065856836736, -2.161648156819865, -0.07871158781927079, -0.01034519518725574, -0.18697667645756155, -1.642529243370518, -0.09579555777600035, -0.15077749220654368, 0.2706500704662176, -1.3203807147219777, -0.381145271967398, -0.5239702265243977, -0.14647357759531587, 1.3052653056802228, 0.0548095490958076],
    'ema_sums': [0.7108612316660583, 0.04334112627839204, -2.192814296344295, -0.07219082713709213, -0.030674481764435768, -0.18185672803156194, -1.5939367623941507, -0.09233932843199, -0.11975255931611173, 0.2817436928889947, -1.37280631554313, -0.3664404007649864, -0.5127080377715174, -0.1654943183530122, 1.29680534475483, 0.05825449201074662],
    'anchor_seed_obj01': 6360478115305869781,
    'update_seed_step2_obj03': 2855451869860620818,
    'cfg_sum_t090': -141.83616635159706, 'cfg_abs_sum_t090': 3091.850153789128,
    'cfg_sum_t050': -51.86031025275588, 'cfg_abs_sum_t050': 1269.7380498237908,
}


def _close(got, want, rel=1e-5, abs_=1e-6):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert math.isclose(g, w, rel_tol=rel, abs_tol=abs_), (got, want)


def _student():
    torch.manual_seed(1235)
    flow = TinyShapeFlow().requires_grad_(False).eval()
    return ST.ShapeStudentNetwork(flow, ST.attach_adapters(flow, 4, 4.0, 0.0))


def _cond(rec):
    coords = rec["coords"].clone()
    coords[:, 0] = 0
    cond = {"global": rec["global"], "proj": Sparse(rec["proj"], coords)}
    return coords, cond, CD.neg_cond_like(cond)


def test_updates_match_original_training_code():
    objects = shape_objects(torch.Generator().manual_seed(99))
    net = _student()
    params = net.trainable()
    opt = O.make_optimizer(params, 1e-2, (0.0, 0.999), 1e-3)
    ema = O.init_ema(params)
    anchors, losses, norms, t_ns, tstd, sstd = {}, [], [], [], [], []
    for step, oid in enumerate(ORDER, start=1):
        coords, cond, neg = _cond(objects[oid])
        if oid not in anchors:
            g = CD.make_generator("cpu", CD.anchor_seed(RUN_SEED, oid))
            eps0 = torch.randn(int(coords.shape[0]), CD.LATENT_CHANNELS, generator=g)
            anchors[oid] = CD.teacher_anchor(net.teacher, Sparse(eps0, coords), cond, neg).feats.detach().clone()
        g = CD.make_generator("cpu", CD.update_seed(RUN_SEED, step - 1, 0, oid))
        n_idx = CD.draw_time_index(g, "cpu")
        t_n, t_next = CD.time_pair(n_idx)
        eps = CD.draw_noise_like(g, anchors[oid])
        sample = CD.Sample(oid, coords, cond, neg, Sparse(anchors[oid], coords), t_n, t_next, eps)
        r = CD.run_update(lambda: CD.consistency_loss(net, net.teacher, params, ema, sample), params, opt, ema, step - 1,
                          grad_clip=10.0, ema_decay=0.99)
        losses.append(r.loss)
        norms.append(r.grad_norm)
        t_ns.append(float(t_n.item()))
        tstd.append(r.info["x0_target_std"])
        sstd.append(r.info["x0_student_std"])
    _close(losses, REFERENCE["losses"])
    _close(norms, REFERENCE["grad_norms"])
    _close(t_ns, REFERENCE["t_n"])
    _close(tstd, REFERENCE["target_std"])
    _close(sstd, REFERENCE["student_std"])
    _close([float(anchors[o].double().sum()) for o in ORDER], [REFERENCE["anchor_sums"][o] for o in ORDER], rel=1e-5, abs_=1e-4)
    _close([float(p.double().sum()) for p in params.values()], REFERENCE["param_sums"])
    _close([float(p.double().sum()) for p in ema.values()], REFERENCE["ema_sums"])


def test_guided_velocity_matches_original_cfg():
    objects = shape_objects(torch.Generator().manual_seed(99))
    net = _student()
    coords, cond, neg = _cond(objects["obj00"])
    g = CD.make_generator("cpu", CD.anchor_seed(RUN_SEED, "obj00"))
    x = Sparse(torch.randn(int(coords.shape[0]), CD.LATENT_CHANNELS, generator=g), coords)
    with torch.no_grad():
        inside = CD.guided_velocity(net.teacher, x, torch.full((1,), 0.9), cond, neg, **CD.GUIDANCE).feats.double()
        outside = CD.guided_velocity(net.teacher, x, torch.full((1,), 0.5), cond, neg, **CD.GUIDANCE).feats.double()
        plain = net.teacher(x, torch.full((1,), 0.5), cond).feats.double()
    _close([float(inside.sum()), float(inside.abs().sum())], [REFERENCE["cfg_sum_t090"], REFERENCE["cfg_abs_sum_t090"]], abs_=1e-4)
    _close([float(outside.sum()), float(outside.abs().sum())], [REFERENCE["cfg_sum_t050"], REFERENCE["cfg_abs_sum_t050"]], abs_=1e-4)
    assert torch.equal(outside, plain)  # outside the interval: the conditional velocity, no guidance


def test_seed_derivation_matches_original():
    assert CD.anchor_seed(RUN_SEED, "obj01") == REFERENCE["anchor_seed_obj01"]
    assert CD.update_seed(RUN_SEED, 1, 0, "obj03") == REFERENCE["update_seed_step2_obj03"]
    assert CD.anchor_seed(RUN_SEED, "obj01") != CD.anchor_seed(RUN_SEED + 1, "obj01")
    assert 0 <= CD.anchor_seed(20260723, "000faf5e58054570bf844bbbc2ed8cf0") < 2 ** 63


def test_time_pairs_and_draws():
    t_n, t_next = CD.time_pair(torch.tensor([1.0]))
    assert math.isclose(float(t_n), 0.98, rel_tol=1e-6) and math.isclose(float(t_next), 0.96, rel_tol=1e-6)
    t_n, t_next = CD.time_pair(torch.tensor([49.0]))
    assert math.isclose(float(t_n), 0.02, rel_tol=1e-5) and 0.0 <= float(t_next) < 1e-6  # float32: ~4e-8, as in the original
    g = CD.make_generator("cpu", 5)
    draws = {int(CD.draw_time_index(g, "cpu").item()) for _ in range(500)}
    assert min(draws) >= 1 and max(draws) <= 49 and len(draws) > 40


def test_ema_ramp():
    assert CD.ema_ramp_decay(0, 0.99) == 0.1
    assert math.isclose(CD.ema_ramp_decay(1, 0.99), 2 / 11)
    assert CD.ema_ramp_decay(889, 0.99) < 0.99 and CD.ema_ramp_decay(890, 0.99) == 0.99 and CD.ema_ramp_decay(5000, 0.99) == 0.99
    p = {"w": torch.nn.Parameter(torch.ones(3))}
    shadow = {"w": torch.zeros(3)}
    d = CD.ema_update_ramp(shadow, p, 0.99, 0)
    assert d == 0.1 and torch.allclose(shadow["w"], torch.full((3,), 0.9))


def test_ema_weights_swap_restores_live_parameters():
    p = {"a": torch.nn.Parameter(torch.ones(2)), "b": torch.nn.Parameter(torch.zeros(2))}
    shadow = {"a": torch.full((2,), 5.0), "b": torch.full((2,), 7.0)}
    with CD.ema_weights(p, shadow):
        assert torch.equal(p["a"].detach(), shadow["a"]) and torch.equal(p["b"].detach(), shadow["b"])
    assert torch.equal(p["a"].detach(), torch.ones(2)) and torch.equal(p["b"].detach(), torch.zeros(2))


def test_neg_cond_like_zeros_global_and_proj():
    coords = torch.zeros(4, 4, dtype=torch.int32)
    cond = {"global": torch.randn(1, 5, 6), "proj": Sparse(torch.randn(4, 12), coords), "note": "kept"}
    neg = CD.neg_cond_like(cond)
    assert torch.equal(neg["global"], torch.zeros(1, 5, 6)) and torch.equal(neg["proj"].feats, torch.zeros(4, 12))
    assert neg["proj"].coords is coords and neg["note"] == "kept"


@pytest.mark.parametrize("n,world,epoch", [(10, 1, 0), (10, 4, 0), (10, 4, 3), (4995, 16, 0), (4990, 16, 7), (17, 16, 1)])
def test_distributed_order_matches_torch_sampler(n, world, epoch):
    for rank in range(world):
        sampler = torch.utils.data.DistributedSampler(list(range(n)), num_replicas=world, rank=rank, shuffle=True,
                                                      seed=RUN_SEED, drop_last=True)
        sampler.set_epoch(epoch)
        assert CD.distributed_order(n, world, rank, RUN_SEED, epoch) == list(iter(sampler))
    with pytest.raises(ValueError):
        CD.distributed_order(3, 4, 0, RUN_SEED, 0)


def test_flow_math_is_pixal3d_convention():
    g = torch.Generator().manual_seed(0)
    x0, eps = torch.randn(10, 32, generator=g), torch.randn(10, 32, generator=g)
    t = torch.tensor([0.37])
    coords = torch.zeros(10, 4, dtype=torch.int32)
    t_tok = CD.expand_t(t, coords)
    x_t = CD.add_noise(x0, eps, t_tok)
    v = (1.0 - CD.SIGMA_MIN) * eps - x0  # Pixal3D's velocity target
    assert torch.allclose(CD.x0_from_v(x_t, v, t_tok), x0, atol=1e-5)
    assert torch.allclose(CD.v_from_x0(x_t.double(), x0.double(), t_tok.double()).float(), v, atol=1e-4)
    assert torch.allclose(CD.euler_step(x_t, v, t, 0.0), x_t - 0.37 * v)
