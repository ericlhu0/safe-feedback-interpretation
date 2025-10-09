"""Tests for the bot avatar PerfectCommunicator."""

from safe_feedback_interpretation.avatars.bot import ArmJointState, PerfectCommunicator


def test_perfect_communicator() -> None:
    """Test PerfectCommunicator with different joint states."""
    a = PerfectCommunicator(ArmJointState([0] * 6))
    assert a.step() == "uncomfortable"
    a.update_state(ArmJointState([45] * 6))
    assert a.step() == "comfortable"
    a.update_state(ArmJointState([15] * 6))
    assert a.step() == "medium"
