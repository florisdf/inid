MAX = 'max'
MIN = 'min'


class RunningExtrema:
    def __init__(self, extremum: str):
        if extremum not in [MAX, MIN]:
            raise ValueError(
                f'Unknown extremum "{extremum}". '
                f'Possible  values: "{MAX}" and "{MIN}".'
            )
        self._extrema_dict = {}
        self.extremum = extremum

    def is_new_extremum(self, key, val):
        if key not in self._extrema_dict:
            try:
                # Check if the value can be compared and returns a (single)
                # boolean value
                if self._comp_fn(val, 0) or True:
                    return True
            except (TypeError, RuntimeError):
                return False

        return self._comp_fn(val, self._extrema_dict[key])

    def _comp_fn(self, new, curr):
        return (
            new >= curr if self.extremum == MAX
            else new <= curr
        )

    def update(self, key, val):
        if self.is_new_extremum(key, val):
            self._extrema_dict[key] = val

    def update_dict(self, d: dict):
        for k, v in d.items():
            self.update(k, v)

    def clear(self):
        self._extrema_dict = {}

    @property
    def extrema_dict(self):
        return {
            f'{self.extremum.title()}{k}': v
            for k, v in self._extrema_dict.items()
        }
