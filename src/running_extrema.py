MAX = 'max'
MIN = 'min'


class RunningExtrema:
    def __init__(self, extremum: str):
        if extremum not in [MAX, MIN]:
            raise ValueError(
                f'Unknown extremum "{extremum}". '
                f'Possible  values: "{MAX}" and "{MIN}".'
            )
        self.extrema_dict = {}
        self.extremum = extremum

    def is_new_extremum(self, key, val):
        if key not in self.extrema_dict:
            try:
                # Check if the value can be compared
                self._comp_fn(val, 0)
                return True
            except ValueError:
                return False

        return self._comp_fn(val, self.extrema_dict[key])

    def _comp_fn(self, new, curr):
        return (
            new >= curr if self.extremum == MAX
            else new <= curr
        )

    def update(self, key, val):
        if self.is_new_extremum(key, val):
            self.extrema_dict[key] = val

    def update_dict(self, d: dict):
        for k, v in d.items():
            self.update(k, v)

    def clear(self):
        self.extrema_dict = {}

    def __dict__(self):
        return {
            f'{self.extremum.title()}{k}': v
            for k, v in self.extrema_dict.items()
        }
