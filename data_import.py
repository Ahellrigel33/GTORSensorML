import numpy as np
import matplotlib.pyplot as plt


class SensorData:
    def __init__(self):
        np.random.seed(2022)
        self.data = dict()
        self.test_data = dict()
        self.imported_labels = ['time_auxdaq_us', 'speed_engine_rpm', 'speed_secondary_rpm', 'lds_pedal_mm', 'imu_acceleration_x', 'imu_acceleration_y', 'imu_acceleration_z', 'pressure_frontbrake_psi']
        self.simple_lables = ['time', 'engine', 'secondary', 'pedal_lds', 'imux', 'imuy', 'imuz', 'frontbrake']

    def import_csv_files(self):
        # Import data from CSV files
        imported = dict()
        imported['cody'] = np.genfromtxt('Cody_4LapTest1.csv', dtype=float, delimiter=",", names=True)
        imported['cody_BIN'] = np.genfromtxt('Cody_4LapTest1_BIN.csv', dtype=float, delimiter=",", names=True)
        imported['abhi'] = np.genfromtxt('Abhi_Test1_BeforeSecondaryBreak.csv', dtype=float, delimiter=",", names=True)
        imported['andrew1'] = np.genfromtxt('Andrew_1.csv', dtype=float, delimiter=",", names=True)
        imported['andrew2'] = np.genfromtxt('Andrew_1Lap_Medium.csv', dtype=float, delimiter=",", names=True)
        imported['andrew3'] = np.genfromtxt('Andrew_2.csv', dtype=float, delimiter=",", names=True)
        imported['andrew3_BIN'] = np.genfromtxt('Andrew_2_BIN.csv', dtype=float, delimiter=",", names=True)
        imported['andrew4_BIN'] = np.genfromtxt('ProbAndrew_BIN.csv', dtype=float, delimiter=",", names=True)
        imported['caden1'] = np.genfromtxt('Caden_3Laps_FullSpeed.csv', dtype=float, delimiter=",", names=True)
        imported['caden1_BIN'] = np.genfromtxt('Caden_3Laps_FullSpeed_BIN.csv', dtype=float, delimiter=",", names=True)
        imported['caden2'] = np.genfromtxt('Caden_3LapTest_FullSpeed.csv', dtype=float, delimiter=",", names=True)

        print(type(imported['cody']))

        # Convert everything to numpy arrays and store sensor data under simpler labels
        for key in imported:
            self.data[key] = dict()
            for i in range(len(self.imported_labels)):
                self.data[key][self.simple_lables[i]] = np.array(imported[key][self.imported_labels[i]])

        # # data_viewer
        # plt.plot(self.data['andrew4_BIN']['time'], self.data['andrew4_BIN']['engine'])
        # plt.show()

    def split_data(self):
        """
        It is important that we separate test data at the very beginning and don't touch until we are ready to
        evaluate our results. Manually selects sections of the data to be used as test data.

        :return:
        """
        for key in self.data:
            print("Key: {}, Size: {}".format(key, self.data[key]['time'].shape))


if __name__ == "__main__":
    S = SensorData()
    S.import_csv_files()
    S.split_data()

