
import copy
import os

from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

def faa_test_func(*args):
	print("Evaluation worked for filename {}!".format(args[5]))

def roll_forward(dataset, model, exp_cfg, revived_cfg, results, filename):

        print("Saving roll forward figures...", flush=True)

        # Fetch segments of appropriate length (feature length + rollforward length)
        shuffled_indices, segment_stack = dataset._load_raw_data(exp_cfg.evaluate.rollforward_length)

        print(f"Segment stack shape: {segment_stack.shape}")

        for i in range(exp_cfg.evaluate.n_rollout_snapshots):

            # Select segment
            selected_segment = segment_stack[shuffled_indices[i]]

            # Split into feature length and rollforward length
            roll_input = selected_segment[0:exp_cfg.dataset.feature_length]
            rollforward_targets = selected_segment[exp_cfg.dataset.feature_length:]

            # rollforward_history holds the feature inputs and predictions
            rollforward_history = roll_input
            rollerror_history = [0]*roll_input.shape[0]

            # Count is equal to length of rollforward length
            count = 0
            current_input = copy.deepcopy(roll_input)
            current_targets = copy.deepcopy(rollforward_targets)

            # Roll forward loop while count isn't finished
            while count < exp_cfg.evaluate.rollforward_length:

                # Feed current inputs
                predictions = model.predict(np.expand_dims(current_input, axis=0))

                # Get forward prediction
                forward_prediction = predictions[1]

                # Append forward prediction into history
                rollforward_history = np.concatenate((rollforward_history, forward_prediction), axis=0)

                # Calculate and append forward prediction error into history
                rollerror_history.append(mse(forward_prediction, current_targets[0]))
                current_targets = current_targets[1:]

                # Pop front from current inputs and insert forward prediction
                current_input = current_input[:5]
                current_input = np.concatenate((current_input, forward_prediction), axis=0)

                # Decrement count
                count += 1

            # Plot and save figure to disk
            full_trajectory = np.concatenate((roll_input, rollforward_targets), axis=0)
            trajectory_plots("forward", i, full_trajectory, rollforward_history, rollerror_history, filename)

        print("Done saving roll forward figures!", flush=True)

def roll_backward(dataset, model, exp_cfg, revived_cfg, results, filename):
    
    print("Saving roll backward figures...", flush=True)

    # Fetch segments of appropriate length (feature length + rollforward length)
    shuffled_indices, segment_stack = dataset._load_raw_data(exp_cfg.evaluate.rollback_length)

    for i in range(exp_cfg.evaluate.n_rollout_snapshots):

        # Select segment
        selected_segment = segment_stack[shuffled_indices[i]]

        # Split into feature length and rollforward length
        segment_length = selected_segment.shape[0]
        roll_input = selected_segment[segment_length-exp_cfg.dataset.feature_length:segment_length]
        rollback_targets = selected_segment[0:segment_length-exp_cfg.dataset.feature_length]

        # rollforward_history holds the feature inputs and predictions
        rollback_history = roll_input
        rollerror_history = [0]*roll_input.shape[0]

        # Count is equal to length of rollforward length
        count = 0
        current_input = copy.deepcopy(roll_input)
        current_targets = copy.deepcopy(rollback_targets)

        # Roll forward loop while count isn't finished
        while count < exp_cfg.evaluate.rollback_length:

            # Feed current inputs
            predictions = model.predict(np.expand_dims(current_input, axis=0))

            # Get forward prediction
            back_prediction = predictions[0]

            # Insert back prediction into history
            rollback_history = np.concatenate((back_prediction, rollback_history), axis=0)

            # Calculate and append forward prediction error into history
            rollerror_history.insert(0, mse(back_prediction, current_targets[current_targets.shape[0]-1]))
            current_targets = current_targets[0:current_targets.shape[0]-1]

            # Pop front from current inputs and insert forward prediction
            current_input = current_input[1:]
            current_input = np.concatenate((back_prediction, current_input), axis=0)

            # Decrement count
            count += 1

        # Plot and save figure to disk
        full_trajectory = np.concatenate((rollback_targets, roll_input), axis=0)
        trajectory_plots("backward", i, full_trajectory, rollback_history, rollerror_history, filename)

    print("Done saving roll backward figures!", flush=True)

def trajectory_plots(direction, snapshot_num, full_trajectory, prediction_history, error_history, filename):
    
    px_val = prediction_history[:, 0]
    py_val = prediction_history[:, 1]
    pz_val = prediction_history[:, 2]

    ax_val = full_trajectory[:, 0]
    ay_val = full_trajectory[:, 1]
    az_val = full_trajectory[:, 2]

    # create figure
    xz_fig = plt.figure()
    ax = xz_fig.add_subplot(1, 1, 1)
    # plot x vs z
    ax.plot(px_val,pz_val, label="Predicted")
    ax.plot(ax_val, az_val, label="Actual")
    plt.title("X vs Z: Actual vs Predicted")
    plt.xlabel("X Value")
    plt.ylabel("Z Value")
    plt.legend(loc='best')
    #plt.show()

    # create figure
    xy_fig = plt.figure()
    ax = xy_fig.add_subplot(1, 1, 1)
    # plot x vs y
    plt.plot(px_val,py_val, label="Predicted")
    plt.plot(ax_val, ay_val, label="Actual")
    plt.title("X vs Y: Actual vs Predicted")
    plt.xlabel("X Value")
    plt.ylabel("Y Value")
    plt.legend(loc='best')
    #plt.show()

    if direction == "forward":
        time = range(0, len(error_history))
    elif direction == "backward":
        time = range(-1*len(error_history), 0, 1)

    #time = range(0, len(error_history))

    # create figure
    err_fig = plt.figure()
    ax = err_fig.add_subplot(1, 1, 1)
    plt.plot(time, error_history)
    ax.set_title("Error")
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position(('data', 0.0))
    ax.spines['bottom'].set_position(('data', 0.0))
    # plot error
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    #plt.show()

    if direction == "forward":
        save_file_path = "{}forward_prediction_plots/".format(filename)
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)

        save_file_path += f"snapshot_{snapshot_num}/"
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)

        xz_fig.savefig(save_file_path + "xz_fig.png", dpi=xz_fig.dpi)
        xy_fig.savefig(save_file_path + "xy_fig.png", dpi=xy_fig.dpi)
        err_fig.savefig(save_file_path + "err_fig.png", dpi=err_fig.dpi)
    if direction == "backward":
        save_file_path = "{}backward_prediction_plots/".format(filename)
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)

        save_file_path += f"snapshot_{snapshot_num}/"
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)
            
        xz_fig.savefig(save_file_path + "xz_fig.png", dpi=xz_fig.dpi)
        xy_fig.savefig(save_file_path + "xy_fig.png", dpi=xy_fig.dpi)
        err_fig.savefig(save_file_path + "err_fig.png", dpi=err_fig.dpi)

def mse(prediction, target):
    return np.sum(np.square(prediction - target))





