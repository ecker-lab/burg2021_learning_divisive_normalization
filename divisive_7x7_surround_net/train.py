import csv
import logging
import os
import time

import matplotlib
import numpy as np

from divisivenormalization.data import Dataset, MonkeySubDataset
from divisivenormalization.models import DivisiveNetOutputNonlin
from divisivenormalization.utils import get_log_hash, log_uniform, time_stamp

matplotlib.use("agg")


def main():
    # prepare saving results into human readable csv
    base_dir = "/projects/burg2021_learning-divisive-normalization/divisive_7x7_surround_net"
    fname = "results.csv"
    param_dict = {
        "run_no": None,
        "avg_corr_after_train": None,
        "stim_size": None,
        "filter_sizes": None,
        "out_channels": None,
        "strides": None,
        "paddings": None,
        "smooth_weights": None,
        "sparse_weights": None,
        "readout_sparse_weight": None,
        "output_nonlin_smooth_weight": None,
        "pool_kernel_size": None,
        "pool_type": None,
        "dilation": None,
        "M": None,
        "dn_exponent_init_val": None,
        "dn_exponent_trainable": None,
        "dn_u_size": None,
        "dn_padding": None,
        "abs_v_l1_weight": None,
        "avg_corr_before_train": None,
        "max_iter": None,
        "batch_size": None,
        "val_steps": None,
        "early_stopping_steps": None,
        "learning_rate": None,
        "lr_decay_factor": None,
        "last_step": None,
        "time_elapsed": None,
        "time_stamp": None,
    }
    csv_fieldnames = param_dict.keys()

    run_no = 0
    # if csv file is available: append, otherwise create new
    if os.path.isfile(os.path.join(base_dir, fname)) is not True:
        with open(os.path.join(base_dir, fname), "w", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=csv_fieldnames)
            writer.writeheader()
    else:
        with open(os.path.join(base_dir, fname), "r", newline="") as csvf:
            reader = csv.DictReader(csvf)
            run_results_list = [dct for dct in reader]
            for d in run_results_list:
                if int(d["run_no"]) >= run_no:
                    run_no = int(d["run_no"]) + 1

    # if log file is not available: create new. Then: append
    log_file_name = "logs.txt"
    if os.path.isfile(os.path.join(base_dir, log_file_name)) is not True:
        with open(os.path.join(base_dir, log_file_name), "w", newline="") as f:
            print("Log file created on" + time_stamp(), file=f)

    # Define model parameters excluded from random hyperparameter search
    filter_sizes = [13]
    out_channels = [32]
    strides = [1]
    paddings = ["VALID"]
    dn_padding = "VALID"
    sparse_weights = [0.0]
    pool_type = "AVG"
    pool_kernel_size = [1, 5, 5, 1]
    dilation = pool_kernel_size
    M = 2
    dn_u_size = [
        7,
        7,
    ]
    abs_v_l1_weight = 0

    learning_rate = 1e-3
    max_iter = 50000
    val_steps = 100
    early_stopping_steps = 10
    batch_size = 256
    eval_batches = 2320

    crop = 0
    data_dict = Dataset.get_clean_data()
    data = MonkeySubDataset(data_dict, seed=1000, train_frac=0.8, subsample=2, crop=crop)
    # Run hyperparameter search
    for _ in range(500):
        start_time = time.time()
        log_hash = get_log_hash(run_no)

        # hyperparameters for random search
        smooth_weights = [log_uniform(low=-9, high=-3.5)]
        readout_sparse_weight = log_uniform(low=-9, high=-4.5)
        output_nonlin_smooth_weight = log_uniform(low=-5, high=0)

        # print logs to terminal
        print("\n")
        print(
            "run_no, smooth_weights, readout_sparse_weight, pool_type, dn_exponent_init_val, abs_v_l1_weight, pool_kernel_size, dilation, dn_u_size, paddings, dn_padding",
            run_no,
            smooth_weights,
            readout_sparse_weight,
            pool_type,
            abs_v_l1_weight,
            pool_kernel_size,
            dilation,
            dn_u_size,
            paddings,
            dn_padding,
        )
        # save logs to file
        with open(os.path.join(base_dir, log_file_name), "a", newline="") as f:
            print(
                "run_no, smooth_weights, readout_sparse_weight, pool_type, dn_exponent_init_val, abs_v_l1_weight, pool_kernel_size, dilation, dn_u_size, paddings, dn_padding",
                run_no,
                smooth_weights,
                readout_sparse_weight,
                pool_type,
                abs_v_l1_weight,
                pool_kernel_size,
                dilation,
                dn_u_size,
                paddings,
                dn_padding,
                file=f,
            )

        try:
            model = DivisiveNetOutputNonlin(
                filter_sizes,
                out_channels,
                strides,
                paddings,
                smooth_weights,
                sparse_weights,
                readout_sparse_weight,
                output_nonlin_smooth_weight,
                pool_kernel_size,
                pool_type,
                dilation,
                M,
                dn_u_size,
                abs_v_l1_weight,
                dn_padding=dn_padding,
                data=data,
                log_dir=os.path.join(base_dir, "train_logs"),
                log_hash=log_hash,
                eval_batches=eval_batches,
            )

            # save first (avoid errors due to loading when no checkpoint has ever been saved)
            model.save()
            model.save_best()

            avg_corr_before_train = model.evaluate_avg_corr_val()

            # train model
            with open(os.path.join(base_dir, log_file_name), "a", newline="") as f:
                print("run Number: " + str(run_no), file=f)
                print(time_stamp(), file=f)

                training = model.train(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    val_steps=val_steps,
                    save_steps=1000,
                    early_stopping_steps=early_stopping_steps,
                    learning_rule_updates=4,
                    eval_batches=eval_batches,
                )
                for (
                    i,
                    (logl, readout_sparse, conv_sparse, smooth, total_loss, pred),
                ) in training:
                    print(
                        "Step %d | Loss: %s | %s: %s | L1 readout: %s | L1 conv: %s | L2 conv: %s | Var(val): %.4f"
                        % (
                            i,
                            total_loss,
                            model.obs_noise_model,
                            logl,
                            readout_sparse,
                            conv_sparse,
                            smooth,
                            np.mean(np.var(pred, axis=0)),
                        )
                    )
                    print(
                        "Step %d | Loss: %s | %s: %s | L1 readout: %s | L1 conv: %s | L2 conv: %s | Var(val): %.4f"
                        % (
                            i,
                            total_loss,
                            model.obs_noise_model,
                            logl,
                            readout_sparse,
                            conv_sparse,
                            smooth,
                            np.mean(np.var(pred, axis=0)),
                        ),
                        file=f,
                    )
                    last_step = i
                print("Done fitting, time elapsed: " + str(time.time() - start_time) + "s")
                print(
                    "Done fitting, time elapsed: " + str(time.time() - start_time) + "s \n",
                    file=f,
                )

            avg_corr_after_train = model.evaluate_avg_corr_val()

        except KeyboardInterrupt:
            return

        except:  # catch all other exceptions, e.g. errors due to non-convergent training etc.
            logging.exception("Error occured.")
            avg_corr_after_train = None
            avg_corr_before_train = None
            last_step = None

        # save parameters and results into csv
        param_dict = {
            "run_no": run_no,
            "avg_corr_after_train": avg_corr_after_train,
            "stim_size": model.data.train()[0].shape[1:3],
            "filter_sizes": filter_sizes,
            "out_channels": out_channels,
            "strides": strides,
            "paddings": paddings,
            "smooth_weights": smooth_weights,
            "sparse_weights": sparse_weights,
            "readout_sparse_weight": readout_sparse_weight,
            "output_nonlin_smooth_weight": output_nonlin_smooth_weight,
            "pool_kernel_size": pool_kernel_size,
            "pool_type": pool_type,
            "dilation": dilation,
            "M": M,
            "dn_exponent_init_val": None,
            "dn_exponent_trainable": None,
            "dn_u_size": dn_u_size,
            "dn_padding": dn_padding,
            "abs_v_l1_weight": abs_v_l1_weight,
            "avg_corr_before_train": avg_corr_before_train,
            "max_iter": max_iter,
            "batch_size": batch_size,
            "val_steps": val_steps,
            "early_stopping_steps": early_stopping_steps,
            "learning_rate": learning_rate,
            "lr_decay_factor": 3,
            "last_step": last_step,
            "time_elapsed": time.time() - start_time,
            "time_stamp": time_stamp(),
        }

        with open(os.path.join(base_dir, fname), "a", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=csv_fieldnames)
            writer.writerow(param_dict)

        for d in param_dict:
            print(d, param_dict[d])

        print("\n\n\n\n\n")

        run_no += 1

    print("Done.")


if __name__ == "__main__":
    print("script started")
    main()
