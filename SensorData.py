import numpy as np
import matplotlib.pyplot as plt
import random
import time


class SensorData:
    def __init__(self, **kwargs):
        np.random.seed(2022)
        self.named_data = dict()
        self.test_data = dict()
        self.validation_data = dict()
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.imported_labels = ['time_auxdaq_us', 'speed_engine_rpm', 'speed_secondary_rpm', 'lds_pedal_mm', 'pressure_frontbrake_psi']
        self.sensors = ['time', 'engine', 'secondary', 'pedal_lds', 'brake']
        self.average_sensors = ['engine', 'secondary']
        self.bin_files = ['cody_BIN', 'andrew3_BIN', 'andrew4_BIN', 'caden1_BIN']
        self.daata_files = ['cody', 'abhi', 'andrew1', 'andrew2', 'andrew3', 'caden1', 'caden2']

        self.holdpoints = kwargs.get('holdpoints', 10)
        self.bin_predict = kwargs.get('bin_predict', 100)  # 1/3 a second
        self.daata_predict = kwargs.get('daata_predict', 7)  # 1/3 a second
        self.bin_average_window = kwargs.get('bin_average_window', 60)
        self.daata_average_window = kwargs.get('daata_average_window', 4)

    def import_csv_files(self):
        # Import data from CSV files
        imported = dict()
        imported['cody'] = np.genfromtxt('CSVFiles/Cody_4LapTest1.csv', dtype=float, delimiter=",", names=True)
        imported['cody_BIN'] = np.genfromtxt('CSVFiles/Cody_4LapTest1_BIN.csv', dtype=float, delimiter=",", names=True)
        imported['abhi'] = np.genfromtxt('CSVFiles/Abhi_Test1_BeforeSecondaryBreak.csv', dtype=float, delimiter=",", names=True)
        imported['andrew1'] = np.genfromtxt('CSVFiles/Andrew_1.csv', dtype=float, delimiter=",", names=True)
        imported['andrew2'] = np.genfromtxt('CSVFiles/Andrew_1Lap_Medium.csv', dtype=float, delimiter=",", names=True)
        imported['andrew3'] = np.genfromtxt('CSVFiles/Andrew_2.csv', dtype=float, delimiter=",", names=True)
        imported['andrew3_BIN'] = np.genfromtxt('CSVFiles/Andrew_2_BIN.csv', dtype=float, delimiter=",", names=True)  # Engine just at steady state
        imported['andrew4_BIN'] = np.genfromtxt('CSVFiles/ProbAndrew_BIN.csv', dtype=float, delimiter=",", names=True)
        imported['caden1'] = np.genfromtxt('CSVFiles/Caden_3Laps_FullSpeed.csv', dtype=float, delimiter=",", names=True)
        imported['caden1_BIN'] = np.genfromtxt('CSVFiles/Caden_3Laps_FullSpeed_BIN.csv', dtype=float, delimiter=",", names=True)
        imported['caden2'] = np.genfromtxt('CSVFiles/Caden_3LapTest_FullSpeed.csv', dtype=float, delimiter=",", names=True)

        # Convert everything to numpy arrays and store sensor data under simpler labels
        for key in imported:
            self.named_data[key] = dict()
            for i in range(len(self.imported_labels)):
                self.named_data[key][self.sensors[i]] = np.array(imported[key][self.imported_labels[i]])

        # # data_viewer
        # plt.plot(self.named_data['andrew4_BIN']['time'], self.named_data['andrew4_BIN']['engine'])
        # plt.show()

    def preprocess_data(self):
        for key in self.bin_files:
            if key in self.named_data.keys():
                for sensor in self.average_sensors:
                    self.named_data[key][sensor] = np.convolve(self.named_data[key][sensor], np.ones(self.bin_average_window, dtype=np.float64)/self.bin_average_window, 'same')
        for key in self.daata_files:
            if key in self.named_data.keys():
                for sensor in self.average_sensors:
                    self.named_data[key][sensor] = np.convolve(self.named_data[key][sensor], np.ones(self.daata_average_window, dtype=np.float64)/self.daata_average_window, 'same')

    def split_data(self):
        """
        It is important that we separate test data at the very beginning and don't touch until we are ready to
        evaluate our results. Manually selects sections of the data to be used as test data.

        :return:
        """
        if 'andrew2' in self.named_data.keys():
            self.test_data['andrew2'] = dict()
        if 'andrew4_BIN' in self.named_data.keys():
            self.test_data['andrew4_BIN'] = dict()
        if 'caden1' in self.named_data.keys():
            self.test_data['caden1'] = dict()
            self.validation_data['caden1'] = dict()
        if 'caden1_BIN' in self.named_data.keys():
            self.test_data['caden1_BIN'] = dict()
            self.validation_data['caden1_BIN'] = dict()
        if 'cody' in self.named_data.keys():
            self.validation_data['cody'] = dict()
        if 'cody_BIN' in self.named_data.keys():
            self.validation_data['cody_BIN'] = dict()

        for key in self.sensors:
            if 'andrew2' in self.named_data.keys():
                self.test_data['andrew2'][key] = self.named_data['andrew2'][key][4000:]
                self.named_data['andrew2'][key] = self.named_data['andrew2'][key][:4000]
            if 'andrew4_BIN' in self.named_data.keys():
                self.test_data['andrew4_BIN'][key] = self.named_data['andrew4_BIN'][key][25000:]
                self.named_data['andrew4_BIN'][key] = self.named_data['andrew4_BIN'][key][:25000]
            if 'caden1' in self.named_data.keys():
                self.test_data['caden1'][key] = self.named_data['caden1'][key][3485:]
                self.validation_data['caden1'][key] = self.named_data['caden1'][key][2324:3485]
                self.named_data['caden1'][key] = self.named_data['caden1'][key][:2324]
            if 'caden1_BIN' in self.named_data.keys():
                self.test_data['caden1_BIN'][key] = self.named_data['caden1_BIN'][key][17938:]
                self.validation_data['caden1_BIN'][key] = self.named_data['caden1_BIN'][key][11959:17938]
                self.named_data['caden1_BIN'][key] = self.named_data['caden1_BIN'][key][:11959]
            if 'cody' in self.named_data.keys():
                self.validation_data['cody'][key] = self.named_data['cody'][key][4000:]
                self.named_data['cody'][key] = self.named_data['cody'][key][:4000]
            if 'cody_BIN' in self.named_data.keys():
                self.validation_data['cody_BIN'][key] = self.named_data['cody_BIN'][key][50000:84000]
                self.named_data['cody_BIN'][key] = np.concatenate((self.named_data['cody_BIN'][key][:50000], self.named_data['cody_BIN'][key][84000:]))

    def print_weights(self, weights):
        index = 0
        string = "    "
        for sensor in self.sensors:
            string += sensor.rjust(12)
        print(string)

        for i in range(-self.holdpoints, 0):
            string = "{}: ".format(i).rjust(5)
            for j in range(len(self.sensors)):
                string += " {:.5f} ".format(weights[index]).rjust(12)
                index += 1
            print(string)

    def print_point(self, point):
        index = 0
        string = "    "
        for sensor in self.sensors:
            string += sensor.rjust(12)
        print(string)

        for i in range(-self.holdpoints, 0):
            string = "{}: ".format(i).rjust(5)
            for j in range(len(self.sensors)):
                string += " {:.2f} ".format(point[index]).rjust(12)
                index += 1
            print(string)

    def aggregate_data(self, percent_data=0.02, use_daata_files=False):
        for k, data_to_aggregate in enumerate([self.named_data, self.validation_data, self.test_data]):
            i = 0
            for key in data_to_aggregate:
                data_to_aggregate[key]['aggregate'] = np.array([data_to_aggregate[key]['time'],
                                                             data_to_aggregate[key]['engine'],
                                                             data_to_aggregate[key]['secondary'],
                                                             data_to_aggregate[key]['pedal_lds'],
                                                            #  data_to_aggregate[key]['imux'],
                                                            #  data_to_aggregate[key]['imuy'],
                                                            #  data_to_aggregate[key]['imuz'],
                                                             data_to_aggregate[key]['brake']]).T

            data_points = 0
            for key in self.bin_files:
                if key in data_to_aggregate.keys():
                    num_points, num_features = data_to_aggregate[key]["aggregate"].shape
                    if k == 0:
                        data_points += int(np.floor(num_points*percent_data))
                    else:
                        data_points += (num_points - self.holdpoints - self.bin_predict)
            if use_daata_files:
                for key in self.daata_files:
                    if key in data_to_aggregate.keys():
                        num_points, num_features = data_to_aggregate[key]["aggregate"].shape
                        if k == 0:
                            data_points += int(np.floor(num_points * percent_data))
                        else:
                            data_points += (num_points - self.holdpoints - self.daata_predict)

            x_data = np.empty((data_points, num_features * self.holdpoints + 1))
            y_data = np.empty((data_points, 2))

            for key in self.bin_files:
                if key in data_to_aggregate.keys():
                    num_points, num_features = data_to_aggregate[key]["aggregate"].shape
                    rng = np.random.default_rng(np.random.randint(10000))  # Make each selection random but still repeatable
                    if k == 0:
                        numbers = rng.choice((num_points - self.holdpoints - self.bin_predict), size=int(np.floor(num_points*percent_data)), replace=False)
                    else:
                        numbers = range((num_points - self.holdpoints - self.bin_predict))
                    for base in numbers:
                        for hp in range(self.holdpoints):
                            x_data[i, hp * num_features:(hp + 1) * num_features] = data_to_aggregate[key]['aggregate'][base + hp, :]
                        x_data[i, num_features * self.holdpoints] = 1

                        y_data[i, 0] = data_to_aggregate[key]['engine'][base + self.holdpoints + self.bin_predict]
                        y_data[i, 1] = data_to_aggregate[key]['secondary'][base + self.holdpoints + self.bin_predict]
                        i += 1

            if use_daata_files:
                for key in self.daata_files:
                    if key in data_to_aggregate.keys():
                        num_points, num_features = data_to_aggregate[key]["aggregate"].shape
                        rng = np.random.default_rng(np.random.randint(10000))  # Make each selection random but still repeatable
                        if k == 0:
                            numbers = rng.choice((num_points - self.holdpoints - self.daata_predict), size=int(np.floor(num_points*percent_data)), replace=False)
                        else:
                            numbers = range((num_points - self.holdpoints - self.daata_predict))

                        for base in numbers:
                            for hp in range(self.holdpoints):
                                x_data[i, hp * num_features:(hp + 1) * num_features] = data_to_aggregate[key]['aggregate'][base + hp, :]
                            x_data[i, num_features * self.holdpoints] = 1

                            y_data[i, 0] = data_to_aggregate[key]['engine'][base + self.holdpoints + self.daata_predict]
                            y_data[i, 1] = data_to_aggregate[key]['secondary'][base + self.holdpoints + self.daata_predict]
                            i += 1
            if k == 0:
                self.x_train = x_data
                self.y_train = y_data
            if k == 1:
                self.x_val = x_data
                self.y_val = y_data
            if k == 2:
                self.x_test = x_data
                self.y_test = y_data
        self.fix_data()

    def noise_injection(self, **kwargs):
        p = kwargs.get('injection_probability', 0.1)
        use_dropout = kwargs.get('use_dropout', False)
        std_dev = np.std(self.x_train, axis=0) * kwargs.get('noise_variation', 0.1)
        print(self.x_train.shape)
        for i in range(self.x_train.shape[0]):
            for j in range(self.x_train.shape[1]):
                if random.random() < p:
                    if use_dropout:
                        self.x_train[i][j] = 0
                    else:
                        self.x_train[i][j] += np.random.normal(0, std_dev[j])

    def fix_data(self):
        data = [self.x_train, self.x_val, self.x_test]
        for k in range(len(data)):
            for j, point in enumerate(data[k]):
                curr_time = point[-(len(self.sensors) + 1)]
                index = 0
                for i in range(self.holdpoints):
                    for sensor in self.sensors:
                        if sensor == 'time':
                            data[k][j][index] -= curr_time
                        if sensor == 'pedal_lds':
                            data[k][j][index] = -(data[k][j][index] - 126)
                        index += 1


if __name__ == "__main__":
    S = SensorData()
    S.import_csv_files()
    S.preprocess_data()
    S.split_data()
    S.aggregate_data()
    #S.noise_injection()

    print(S.print_point(S.x_train[600]))
    print(S.print_point(S.x_train[800]))

    # print(S.x_train.shape)
    # print(np.sum(np.isnan(S.x_train)))
    # print(S.x_train.dtype)
    #
    # print(S.y_train.shape)

    # plt.figure()
    # plt.plot(S.x_train[:, -8])
    # plt.plot(S.x_train[:, -7])
    # plt.figure()
    # plt.plot(S.y_train)
    # plt.show()

