"""A smaller denominator must not turn selecting noise into genuine gain."""
import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.reward_noise_control import independent_terminal_correction,noise_control


def test_terminal_correction_cancels_selected_search_gain_and_keeps_signed_loss():
    correction=independent_terminal_correction(search_gain=np.array([.2]),
        confirmation_initial=np.array([[1.,1.]]),confirmation_selected=np.array([[1.1,1.1]]),scale=.1)
    assert (.2/.1+correction)[0]==pytest.approx(-1.)


@pytest.mark.parametrize('sigma',[.0001,.001,.01])
def test_noise_only_control_rejects_naive_scaling_but_not_independent_gain(sigma):
    result=noise_control(seed=20261102,sigma=sigma)
    assert result['naive']['descriptive_mean_interval'][0]>0
    assert result['corrected']['descriptive_mean_interval'][1]<0
    assert result['max_queries']==8 and result['new_model_calls']==0


def test_signal_can_pay_for_queries_without_requiring_reward_to_be_positive():
    result=noise_control(seed=20261102,sigma=.001,signal=.002)
    assert result['latent_mean_improvement']>0
    assert result['corrected']['descriptive_mean_interval'][0]>0
