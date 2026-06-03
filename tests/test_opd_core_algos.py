import pytest

torch = pytest.importorskip("torch")

from verl.agent_trainer.ppo import core_algos

schemas = pytest.importorskip("verl.workers.rollout.schemas")
RolloutHandler = schemas.RolloutHandler


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


def test_rollout_action_mask_excludes_assistant_suffix_token():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == "\n<|im_start|>assistant\n":
                return [100]
            if text == "<|im_end|>":
                return [101]
            return [ord(ch) for ch in text]

        def decode(self, ids):
            return str(ids)

    handler = RolloutHandler(
        messages=[],
        task_name="dummy",
        item_id=0,
        score=0.0,
        done=False,
        input_ids=[101],
        prompt_ids=[101],
        response_ids=[],
        attention_mask=[1],
        prompt_attention_mask=[1],
        response_attention_mask=[],
        position_ids=[0],
        prompt_position_ids=[0],
        response_position_ids=[],
        loss_mask=[0],
        prompt_loss_mask=[0],
        response_loss_mask=[],
        max_response_len=8,
        max_model_len=16,
    )

    handler.add_assistant_message(FakeTokenizer(), "ab")

    assert handler.loss_mask == [0, 0, 1, 1, 1]
    assert handler.action_mask == [0, 0, 1, 1, 0]

    handler.truncate_output_ids()

    assert handler.response_loss_mask == [0, 1, 1, 1]
    assert handler.response_action_mask == [0, 1, 1, 0]
