from src.database.Models import sameClassSensors, findNearestSensors


def same_class_sensors_t():
    tclass_sids = sameClassSensors('residential')
    neighbors = findNearestSensors(11535, tclass_sids, n=100)
    print(len(neighbors))


def run_tests():
    same_class_sensors_t()


if __name__ == '__main__':
    run_tests()