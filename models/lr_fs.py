import tensorflow as tf
from absl import logging

from sklearn.linear_model import LogisticRegression
from models import register_model, BaseModel, np


@register_model("lr_fs")
class LogisticRegressionFewShotModel(BaseModel):
    def __init__(self, base_dir, model_id, params):
        """
        SKLearn logistic regression head used for meta-test time during few-shot classification.
        :param base_dir:
        :param model_id:
        :param params:
        """
        super(LogisticRegressionFewShotModel, self).__init__(base_dir, model_id)

    def episode(self, model, dataset, episodes):
        way, shot = dataset.way, dataset.shot
        running_acc = tf.keras.metrics.Mean()
        acc_list = []

        episode = 0
        for example in dataset.load(repeat=True):
            (s_x_image, s_class_ids), (q_x_image, q_class_ids) = example[:len(example) // 2][-2:], example[len(example) // 2:][-2:]
            if episodes and episode == episodes:
                break

            episode += 1

            # Splits reduce batch size and memory requirement when running our embedding model.
            SPLITS = 2
            mu = tf.concat([model.embed(x)[1] for x in tf.split(tf.concat((s_x_image, q_x_image), axis=0), SPLITS, axis=0)], axis=0)
            embeddings = mu

            embeddings, _ = tf.linalg.normalize(embeddings, axis=1)

            support_embeddings = embeddings[:way * shot].numpy()
            query_embeddings = embeddings[way * shot:].numpy()

            _, indices = np.unique(np.concatenate((s_class_ids.numpy(), q_class_ids.numpy())), return_inverse=True)
            support_labels, query_labels = indices[:len(support_embeddings)], indices[len(support_embeddings):]

            lr_model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
            lr_model.fit(support_embeddings, support_labels)
            acc = lr_model.score(query_embeddings, query_labels)

            running_acc(acc)
            acc_list.append(acc)
            if episode % 500 == 0:
                logging.info("Episode: %d | Accuracy: %.4f | Running Acc: %.4f", episode, acc, running_acc.result())

        logging.info("Final Result | Mean Accuracy: %.4f | Std: %.4f | Var: %.4f | p95: %.4f",
                     running_acc.result(), np.std(acc_list), np.var(acc_list), 1.96 * np.std(acc_list) / np.sqrt(len(acc_list)))
        # logging.info("Few-shot evaluation with Logistic Regression, Complete.")