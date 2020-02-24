from absl.testing import absltest

import learner_server

class LearnerServerTest(absltest.TestCase):

  def test_grpc_server(self):
    """ddd"""
    learner = learner_server.setup_learner()
    server = learner_server.setup_server(learner)
    print(learner)
    self.assertEqual(learner.is_done(), False)
    print(server)
    server.start()
