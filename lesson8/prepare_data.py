import os

import paddle.v2.dataset as dataset

# NOTE: You should full fill your username, for example:
#   USERNAME = "paddle@example.com"
# TODO(Yancey1989): fetch username from environment variable.
USERNAME = "tanzhongyi@baidu.com"

DC = os.getenv("PADDLE_CLOUD_CURRENT_DATACENTER")

# PaddleCloud cached the dataset on /pfs/${DATACENTER}/home/${USERNAME}/...
dataset.common.DATA_HOME = "/pfs/%s/home/%s" % (DC, USERNAME)
TRAIN_FILES_PATH = os.path.join(dataset.common.DATA_HOME, "movielens")

TRAINER_ID = int(os.getenv("PADDLE_INIT_TRAINER_ID"))
TRAINER_INSTANCES = int(os.getenv("PADDLE_INIT_NUM_GRADIENT_SERVERS"))


def main():

    if TRAINER_ID == -1 or TRAINER_INSTANCES == -1:
        print "no cloud environ found, must run on cloud"
        exit(1)

    print("\nBegin to convert data into "+ dataset.common.DATA_HOME)
    dataset.common.convert(TRAIN_FILES_PATH,
                           dataset.movielens.train(), 1000, "train")
    print("\nConvert process is finished")
    print("\nPlease run 'paddlecloud file ls "+ dataset.common.DATA_HOME+
          "/movielens' to check if datas exist there")


if __name__ == '__main__':
    main()
