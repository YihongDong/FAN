import numpy as np

def sawtooth_wave(t, n):
    """Generate a single term of the sawtooth wave harmonic series."""
    return (t / np.pi) - np.floor(t / np.pi + 0.5)

def gen_periodic_data(periodic_type):

    if periodic_type == 'sin':
        def generate_periodic_data(num_samples, num_periods=100, is_train = True):
            if is_train:
                t = np.linspace(-num_periods * np.pi, num_periods * np.pi, num_samples)
            else:
                t = np.linspace(-num_periods * 3 * np.pi, num_periods * 3 * np.pi, num_samples)
            data = np.sin(t)  # 使用正弦函数
            return t, data
        print(f'generate data from the {periodic_type} function')

        PERIOD = 6
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD), PERIOD)
        t_test, data_test = generate_periodic_data(4000, PERIOD, is_train = False)

        y_uper = 1.5
        y_lower = -1.5
    
    # ----------------------------------------------------------------------------------------------------------
    
    elif periodic_type == 'mod':
        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples) 
            data = [i%5 for i in t]
            return t, data

        print(f'generate data from the {periodic_type} function')

        PERIOD = 20
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 10
        y_lower = -5
    

    # ----------------------------------------------------------------------------------------------------------

    elif periodic_type == 'complex_1':

        # complex_period
        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples)
            data = np.exp(np.sin(np.pi * t)**2 + np.cos(t) + t%3 - 1)
            return t, data
        print(f'generate data from the {periodic_type} function')

        PERIOD = 4
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 20
        y_lower = -20        
    
    # ----------------------------------------------------------------------------------------------------------

    elif periodic_type == 'complex_2':
        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples)   

            data = (1 + np.sin(t)) * np.sin(2 * t)
            return t, data
        print(f'generate data from the {periodic_type} function')

        PERIOD = 4
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 4
        y_lower = -4

    # ----------------------------------------------------------------------------------------------------------

    elif periodic_type == 'complex_3':

        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples)   

            data = np.sin(t + np.sin(2 * t))
            return t, data
        print(f'generate data from the {periodic_type} function')

        PERIOD = 4
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 2
        y_lower = -2

    # ----------------------------------------------------------------------------------------------------------

    elif periodic_type == 'complex_4':

        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples)   

            data = np.sin(t) * np.cos(2 * t)**2 + np.cos(t) * np.sin(3 * t)**2
            return t, data
        print(f'generate data from the {periodic_type} function')

        PERIOD = 4
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 2
        y_lower = -2


    # ----------------------------------------------------------------------------------------------------------

    elif periodic_type == 'complex_5':

        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples) 

            N = 5
            data = np.zeros_like(t)
            for n in range(1, N+1):
                data += (1/n) * sawtooth_wave(n * t, n)

            return t, data
        print(f'generate data from the {periodic_type} function')

        PERIOD = 4
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 1
        y_lower = -1

    # ----------------------------------------------------------------------------------------------------------

    elif periodic_type == 'complex_6':

        def generate_periodic_data(num_samples, PERIOD=100, is_train = True):
            if is_train:
                t = np.linspace(-PERIOD, PERIOD, num_samples)
            else:
                t = np.linspace(-2*PERIOD, 2*PERIOD, num_samples) 

            data = np.exp(np.sin(t)) / (1 + np.cos(2 * t)**2)

            return t, data
        print(f'generate data from the {periodic_type} function')

        PERIOD = 4
        BATCHSIZE = 32
        NUMEPOCH = 10000
        PRINTEPOCH = 50
        lr = 1e-5
        wd = 0.01

        t, data = generate_periodic_data(int(10000*PERIOD))
        t_test, data_test = generate_periodic_data(4000, is_train = False)

        y_uper = 3
        y_lower = 0


    return t, data, t_test, data_test, PERIOD, BATCHSIZE, NUMEPOCH, PRINTEPOCH, lr, wd, y_uper, y_lower


def plot_periodic_data(t, data, t_test, data_test, result, args, epoch, path, y_uper, y_lower):
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(35, 5))
    plt.plot(t_test, data_test, label='Domain of Test Data', color='blue')
    plt.plot(t, data, label='Domain of Training Data', color='green')
    plt.plot(t_test, result, label='Model Predictions', color='red', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(min(t_test),max(t_test))
    plt.ylim(y_lower, y_uper)
    # plt.legend()
    plt.savefig(f'{path}/epoch{epoch}.png')
    
def read_log_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        train_loss = []
        test_loss = []
        for line in lines:
            if 'Train Loss' in line:
                train_loss.append(float(line.split(' ')[-1].strip()))
            elif 'Test Loss' in line:
                test_loss.append(float(line.split(' ')[-1].strip()))
    return train_loss, test_loss

def plot_periodic_loss(log_file_path):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    train_log_loss, test_log_loss = read_log_file(log_file_path)
    
    log_file_name = log_file_path.split('.')[0]
    ax1.plot(np.arange(0,len(train_log_loss)*50,50),train_log_loss, label=log_file_name)
    ax2.plot(np.arange(0,len(test_log_loss)*50,50),test_log_loss, label=log_file_name)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend(loc='upper right')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Loss')
    ax2.legend(loc='upper right')
    plt.savefig(f'{log_file_name}.pdf')