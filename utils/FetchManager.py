class FetchManager:
    def __init__(self, sess, fetches):
        self.fetches = fetches
        self.sess = sess

    def fetch(self, feed_dictionary, additional_fetches=[]):
        fetches = self.fetches + additional_fetches
        evaluation = self.sess.run(fetches, feed_dictionary)
        return {k:v for k,v in zip(fetches, evaluation)}