import numpy as np
import matplotlib.pyplot as plt
import math


class SensorData:
    def __init__(self, **kwargs):
        np.random.seed(2022)
        self.named_data = dict()
        self.test_data = dict()
        self.x_train = list()
        self.y_train = list()
        self.imported_labels = ['time_auxdaq_us', 'speed_engine_rpm', 'speed_secondary_rpm', 'lds_pedal_mm', 'imu_acceleration_x', 'imu_acceleration_y', 'imu_acceleration_z', 'pressure_frontbrake_psi']
        self.sensors = ['time', 'engine', 'secondary', 'pedal_lds', 'imux', 'imuy', 'imuz', 'front_brake']
        self.average_sensors = ['engine', 'secondary', 'imux', 'imuy', 'imuz']
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
        imported['andrew3_BIN'] = np.genfromtxt('CSVFiles/Andrew_2_BIN.csv', dtype=float, delimiter=",", names=True)
        imported['andrew4_BIN'] = np.genfromtxt('CSVFiles/ProbAndrew_BIN.csv', dtype=float, delimiter=",", names=True)
        imported['caden1'] = np.genfromtxt('CSVFiles/Caden_3Laps_FullSpeed.csv', dtype=float, delimiter=",", names=True)
        imported['caden1_BIN'] = np.genfromtxt('CSVFiles/Caden_3Laps_FullSpeed_BIN.csv', dtype=float, delimiter=",", names=True)
        imported['caden2'] = np.genfromtxt('CSVFiles/Caden_3LapTest_FullSpeed.csv', dtype=float, delimiter=",", names=True)

        print(type(imported['cody']))

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
            for sensor in self.average_sensors:
                self.named_data[key][sensor] = np.convolve(self.named_data[key][sensor], np.ones(self.bin_average_window)/self.bin_average_window, 'same')
        for key in self.daata_files:
            for sensor in self.average_sensors:
                self.named_data[key][sensor] = np.convolve(self.named_data[key][sensor], np.ones(self.daata_average_window)/self.daata_average_window, 'same')

    def split_data(self):
        """
        It is important that we separate test data at the very beginning and don't touch until we are ready to
        evaluate our results. Manually selects sections of the data to be used as test data.

        :return:
        """
        self.test_data['andrew2'] = dict()
        self.test_data['andrew4_BIN'] = dict()
        self.test_data['caden1'] = dict()
        self.test_data['caden1_BIN'] = dict()
        for key in self.sensors:
            self.test_data['andrew2'][key] = self.named_data['andrew2'][key][4000:]
            self.named_data['andrew2'][key] = self.named_data['andrew2'][key][:4000]
            self.test_data['andrew4_BIN'][key] = self.named_data['andrew4_BIN'][key][25000:]
            self.named_data['andrew4_BIN'][key] = self.named_data['andrew4_BIN'][key][:25000]
            self.test_data['caden1'][key] = self.named_data['caden1'][key][3485:]
            self.named_data['caden1'][key] = self.named_data['caden1'][key][:3485]
            self.test_data['caden1_BIN'][key] = self.named_data['caden1_BIN'][key][17938:]
            self.named_data['caden1_BIN'][key] = self.named_data['caden1_BIN'][key][:17938]

        if __name__ == "__main__":
            for key in self.named_data:
                print("Key: {}, Size: {}".format(key, self.named_data[key]['time'].shape))
            for key in self.test_data:
                print("Key: {}, Size: {}".format(key, self.test_data[key]['time'].shape))

    def aggregate_data(self, percent_data=0.02):
        for key in self.named_data:
            self.named_data[key]['aggregate'] = np.array([self.named_data[key]['time'],
                                                         self.named_data[key]['engine'],
                                                         self.named_data[key]['secondary'],
                                                         self.named_data[key]['pedal_lds'],
                                                         self.named_data[key]['imux'],
                                                         self.named_data[key]['imuy'],
                                                         self.named_data[key]['imuz'],
                                                         self.named_data[key]['front_brake']]).T

        data_points = 0
        for key in self.bin_files:
            num_points, num_features = self.named_data[key]["aggregate"].shape
            data_points += math.floor(num_points*percent_data)

        self.x_train = np.empty((data_points, num_features * self.holdpoints + 1))
        self.y_train = np.empty((data_points, 2))

        for key in self.bin_files:
            num_points, num_features = self.named_data[key]["aggregate"].shape
            rng = np.random.default_rng(np.random.randint(10000))  # Make each selection random but still repeatable
            numbers = rng.choice((num_points - self.holdpoints - self.bin_predict), size=math.floor(num_points*percent_data), replace=False)

            for i, base in enumerate(numbers):
                for hp in range(self.holdpoints):
                    self.x_train[i, hp * num_features:(hp + 1) * num_features] = self.named_data[key]['aggregate'][base + hp, :]
                self.x_train[i, num_features * self.holdpoints] = 1

                self.y_train[i, 0] = self.named_data[key]['engine'][base + self.holdpoints + self.bin_predict]
                self.y_train[i, 1] = self.named_data[key]['secondary'][base + self.holdpoints + self.bin_predict]



if __name__ == "__main__":
    S = SensorData()
    S.import_csv_files()
    S.preprocess_data()
    S.split_data()
    S.aggregate_data()

    print(S.x_train.shape)

