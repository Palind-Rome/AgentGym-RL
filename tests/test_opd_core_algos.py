import pytest

torch = pytest.importorskip("torch")

from verl.agent_trainer.ppo import core_algos


def test_sampled_token_opd_uses_response_mask():
    old_log_prob = torch.tensor([[-2.0, -1.0, -4.0]])
    log_prob = old_log_prob.clone().requires_grad_(True)
    teacher_log_prob = torch.tensor([[-1.0, -3.0, -2.0]])
    response_mask = torch.tensor([[1.0, 0.0, 1.0]])

    loss, metrics = core_algos.compute_sampled_token_distillation_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        teacher_log_prob=teacher_log_prob,
        response_mask=response_mask,
        cliprange=0.2,
        loss_mode='pg_reverse_kl')

    assert loss.ndim == 0
    assert 'distillation/loss' in metrics

    loss.backward()
    assert log_prob.grad is not None
    assert log_prob.grad[0, 1].item() == 0.0
    assert log_prob.grad[0, 0].item() < 0.0
    assert log_prob.grad[0, 2].item() < 0.0


def test_sampled_token_opd_mse_mode_is_scalar():
    old_log_prob = torch.tensor([[-2.0, -1.0]])
    log_prob = old_log_prob.clone().requires_grad_(True)
    teacher_log_prob = torch.tensor([[-1.5, -0.5]])
    response_mask = torch.tensor([[1.0, 1.0]])

    loss, metrics = core_algos.compute_sampled_token_distillation_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        teacher_log_prob=teacher_log_prob,
        response_mask=response_mask,
        cliprange=0.2,
        loss_mode='mse')

    assert loss.ndim == 0
    assert metrics['distillation/loss'] == loss.detach().item()
