import numpy as np
import argparse
import logging
import os
import sys
import csv
import s3fs
import json

import tensorflow as tf
from transformers import AutoTokenizer, RobertaConfig, TFAutoModelForSequenceClassification


fs = s3fs.S3FileSystem()
MAX_LEN = 128
PEACE_COUNTRY = set(['Australia', 'New Zealand', 
                 'Belgium', 'Sweden', 'Denmark', 
                 'Norway', 'Finland', 'Czech Republic', 
                 'Netherlands', 'Austria'])

def regular_encode(texts, tokenizer, maxlen=MAX_LEN):
    """
    Function to encode the word
    """
    # encode the word to vector of integer
    enc_di = tokenizer.encode_plus(
        texts, 
        return_attention_mask=True, 
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        max_length=maxlen)
    
    return np.array(enc_di['input_ids']), np.array(enc_di['attention_mask'])


def read_csv(file_path = 's3://compressed-data-sample/processed_train.json'):
    for count, line in enumerate(fs.open(file_path)):
        if count >= 1e4:
            return
        json_file = json.loads(line)
        ids, msk = regular_encode(json_file['content_cleaned'], tokenizer) # tokenize content_cleaned
        yield {'input_ids': ids,'attention_mask':msk}, int(json_file['country'] in PEACE_COUNTRY)

def is_valid(x, _):
    return x % 10 <= 1

def is_train(x, y):
    return not is_valid(x, y)

recover = lambda x, y: y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_eval", type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = RobertaConfig.from_pretrained(
        args.model_name,
        num_labels=1, #Binary Classification
        dropout=0.1,
        attention_dropout=0.1,
        output_hidden_states=False,
        output_attentions=False
    )

    bert_model = TFAutoModelForSequenceClassification.from_pretrained(args.model_name, trainable=True, config=config)
    input_ids_in = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_ids', dtype='int32')
    input_masks_ids_in = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention_mask', dtype='int32')
    output_layer = bert_model(input_ids_in, input_masks_ids_in)[0]
    output_layer = tf.keras.layers.Activation(activation='sigmoid')(output_layer)
    model = tf.keras.Model(inputs=[input_ids_in, input_masks_ids_in], outputs = output_layer)
    
    # fine optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
               tf.keras.metrics.Precision(name='precision', thresholds=0.5),
               tf.keras.metrics.Recall(name='recall', thresholds=0.5)]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Load Input Dataset
    ds = tf.data.Dataset.from_generator(read_csv, ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int16))
    # Split the dataset for validation.
    tf_train_dataset = ds.enumerate()\
                         .filter(is_train).map(recover)\
                         .cache()\
                         .shuffle(args.train_batch_size)\
                         .batch(args.train_batch_size)\
                         .prefetch(tf.data.AUTOTUNE)
    # Split the dataset for training.
    tf_valid_dataset = ds.enumerate()\
                         .filter(is_valid).map(recover)\
                         .cache()\
                         .shuffle(args.eval_batch_size)\
                         .batch(args.eval_batch_size)\
                         .prefetch(tf.data.AUTOTUNE)
    
    
    # Training
    if args.do_train:

        train_results = model.fit(tf_train_dataset, epochs=args.epochs, batch_size=args.train_batch_size)
        logger.info("*** Train ***")

        output_eval_file = os.path.join(args.output_data_dir, "train_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Train results *****")
            logger.info(train_results)
            for key, value in train_results.history.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
                
   # Evaluation
    if args.do_eval:

        result = model.evaluate(tf_valid_dataset, batch_size=args.eval_batch_size, return_dict=True)
        logger.info("*** Evaluate ***")

        output_eval_file = os.path.join(args.output_data_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info(result)
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

    # Save result
    bert_model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)