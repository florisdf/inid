from recognite.utils import RunningExtrema, MAX, MIN


def test_is_new_max():
    running_max = RunningExtrema(MAX)

    running_max.update_dict({
        'A': 10,
        'B': 500
    })

    assert running_max.is_new_extremum('A', 1000)
    assert not running_max.is_new_extremum('B', 400)


def test_update_dict_max():
    running_max = RunningExtrema(MAX)

    running_max.update_dict({
        'A': 10,
        'B': 500
    })
    running_max.update_dict({
        'A': 1000,
        'B': 400
    })

    assert running_max.extrema_dict['A'] == 1000
    assert running_max.extrema_dict['B'] == 500


def test_is_new_min():
    running_min = RunningExtrema(MIN)

    running_min.update_dict({
        'A': 10,
        'B': 500
    })

    assert not running_min.is_new_extremum('A', 1000)
    assert running_min.is_new_extremum('B', 400)


def test_update_dict_min():
    running_min = RunningExtrema(MIN)

    running_min.update_dict({
        'A': 10,
        'B': 500
    })
    running_min.update_dict({
        'A': 1000,
        'B': 400
    })

    assert running_min.extrema_dict['A'] == 10
    assert running_min.extrema_dict['B'] == 400


def test_clear():
    running_max = RunningExtrema(MAX)

    running_max.update_dict({
        'A': 10,
        'B': 500
    })
    running_max.clear()

    assert len(running_max.extrema_dict) == 0


def test_update_max():
    running_max = RunningExtrema(MAX)

    running_max.update('A', 10)
    running_max.update('B', 500)

    assert running_max.extrema_dict['A'] == 10
    assert running_max.extrema_dict['B'] == 500


def test_update_min():
    running_min = RunningExtrema(MIN)

    running_min.update('A', 10)
    running_min.update('B', 500)

    assert running_min.extrema_dict['A'] == 10
    assert running_min.extrema_dict['B'] == 500


def test_iterable_not_added():
    running_max = RunningExtrema(MAX)
    running_max.update('A', [0, 1, 2])
    assert len(running_max.extrema_dict) == 0
