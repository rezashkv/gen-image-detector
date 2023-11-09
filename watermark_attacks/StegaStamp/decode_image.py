import bchlib
import glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow.compat.v1 as tf
# import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from sklearn import metrics
from tqdm import tqdm

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--org_images_dir', type=str, default='path_to_image_dataset')
    # parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--secret', type=str, required=True)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    
    org_files_list = glob.glob(args.org_images_dir + '/*')

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    
    secret_bits = []
    [[secret_bits.append((byte >> i) % 2) for i in range(7,-1,-1)] for byte in bytearray(args.secret + ' '*(7-len(args.secret)), 'utf-8')]

    def get_scores(files_list, N=100):
        scores = []
        for i_f, filename in tqdm(enumerate(files_list)):
            if i_f >= N:
                break
            image = Image.open(filename).convert("RGB")
            image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
            image /= 255.

            feed_dict = {input_image:[image]}

            secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]

            packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
            packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
            packet = bytearray(packet)

            data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

            # bitflips = bch.decode_inplace(data, ecc)

            data_bits = []
            [[data_bits.append((byte >> i) % 2) for i in range(7,-1,-1)] for byte in data]



            if len(data_bits) != len(secret_bits):
                print("ERORRRRRR")
                print(data)
                exit()

            score = sum([data_bits[i] == secret_bits[i] for i in range(len(data_bits))]) / len(data_bits)
            scores.append(score)
            # print(score)
        return scores

    data_cnt = 64

    org_scores = get_scores(org_files_list, data_cnt)
    wm_scores = get_scores(files_list, data_cnt)

    labels = [0 for i in range(len(org_scores))] + [1 for i in range(len(wm_scores))]
    preds = org_scores + wm_scores

    auroc = metrics.roc_auc_score(labels, preds)

    print(org_scores)
    print(wm_scores)
    print(f"AVG ORG: {(sum(org_scores) / len(org_scores))}, AVG WM: {(sum(wm_scores) / len(wm_scores))}")
    print("AUROC:", auroc)

if __name__ == "__main__":
    main()
