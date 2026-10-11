import itertools
import unittest
import torch
import policy_gradient_demo as pg


class Tests(unittest.TestCase):
    def test_returns_and_gamma_edges(self):
        r = torch.tensor([1., 2., 10.])
        for gamma, expected in [(0., [1., 2., 10.]), (1., [13., 12., 10.]),
                                (.9, [10.9, 11., 10.])]:
            torch.testing.assert_close(pg.discounted_returns(r, gamma), torch.tensor(expected))

    def test_actor_gradient_isolation_and_critic_update(self):
        for method in range(1, 7):
            actor, v, q = pg.Actor(), pg.ValueCritic(), pg.QCritic()
            b = pg.rollout(actor, pg.ToyEnv(), .9)
            weights, critic_loss = pg.METHODS[method-1](b, v, q, .9)
            pg.actor_loss(actor, b, weights).backward()
            self.assertTrue(any(p.grad is not None for p in actor.parameters()))
            self.assertTrue(all(p.grad is None for n in (v, q) for p in n.parameters()))
            if critic_loss is not None:
                critic_loss.backward()
                self.assertTrue(any(p.grad is not None for n in (v, q) for p in n.parameters()))

    def test_method_weights_and_targets(self):
        actor = pg.Actor()
        b = pg.rollout(actor, pg.ToyEnv(), .9)
        b.rewards = torch.tensor([1., 2., 10.])
        b.returns = pg.discounted_returns(b.rewards, .9)
        v = lambda states: states @ torch.tensor([8., 6., 4.])
        q = lambda states, actions: states @ torch.tensor([10., 7., 9.])
        expected = [[10.9,10.9,10.9], [10.9,9.9,8.1], [2.9,4.5,4.86],
                    [10.,6.3,7.29], [2.,.9,4.05], [-1.6,-.36,4.86]]
        for i, method in enumerate(pg.METHODS):
            weights, loss = method(b,v,q,.9)
            torch.testing.assert_close(weights,torch.tensor(expected[i]))
            target = pg.td_target(b,v,.9) if i == 5 else b.returns
            terms = []
            if i in (2,4,5):
                terms.append(.5*(v(b.states)-target).square().mean())
            if i in (3,4):
                terms.append(.5*(q(b.states,b.actions)-target).square().mean())
            if terms:
                torch.testing.assert_close(loss,sum(terms))
            else:
                self.assertIsNone(loss)

    def test_terminal_mask_and_target_detach(self):
        actor, v = pg.Actor(), pg.ValueCritic()
        b = pg.rollout(actor, pg.ToyEnv(), .9)
        b.next_states[-1] = float('nan')  # Terminal observation must not be evaluated.
        y = pg.td_target(b, v, .9)
        self.assertFalse(y.requires_grad)
        self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(y[-1], b.rewards[-1])
        torch.testing.assert_close(y[:-1], b.rewards[:-1]+.9*v(b.next_states[:-1]))

    def test_reject_incomplete_mc(self):
        class Truncated(pg.ToyEnv):
            def step(self, a):
                s,r,_,_ = super().step(a)
                return s,r,False,True
        with self.assertRaises(ValueError):
            pg.rollout(pg.Actor(), Truncated(), .9)

    def test_exact_expected_gradients_all_six(self):
        # Enumerate all trajectories of a two-step, action-dependent MDP.
        # State at t=1 depends on a0; exact V/Q for this policy are constants
        # in each score-function estimator. Compare against analytic autograd J.
        for gamma in (0., .4, 1.):
            theta = torch.tensor([.2, -.4, .6], requires_grad=True)
            p = theta.sigmoid()
            v1 = torch.stack([2*p[1], -1+3*p[2]])
            q0 = torch.stack([gamma*v1[0], 1+gamma*v1[1]])
            v0 = (1-p[0])*q0[0]+p[0]*q0[1]
            exact = torch.autograd.grad(v0, theta, retain_graph=True)[0]
            estimates = [torch.zeros(3) for _ in range(6)]
            for a0,a1 in itertools.product((0,1), repeat=2):
                logp = torch.stack([torch.log(p[0] if a0 else 1-p[0]),
                                   torch.log(p[1+a0] if a1 else 1-p[1+a0])])
                prob = logp.sum().exp().detach()
                r = torch.tensor([float(a0), float(2*a1 if a0==0 else -1+3*a1)])
                G = pg.discounted_returns(r,gamma)
                V = torch.stack([v0,v1[a0]]).detach()
                Q = torch.stack([q0[a0],r[1]]).detach()
                d = torch.tensor([1., gamma])
                td = r + gamma*torch.stack([V[1],torch.tensor(0.)])-V
                weights = [G[0].expand(2),d*G,d*(G-V),d*Q,d*(Q-V),d*td]
                for i,w in enumerate(weights):
                    estimates[i] += prob*torch.autograd.grad((w.detach()*logp).sum(),theta,retain_graph=True)[0]
            for est in estimates:
                torch.testing.assert_close(est, exact, atol=1e-6, rtol=1e-5)

    def test_training_all_methods(self):
        for method in range(1,7):
            actor,_,_ = pg.train(method,episodes=300,verbose=False)
            p = actor(torch.eye(3)).probs[:,1].mean().item()
            print(f'method {method}: P(action=1)={p:.4f}')
            self.assertGreater(p,.8)


if __name__=='__main__':
    unittest.main()
