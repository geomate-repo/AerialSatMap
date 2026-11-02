import torch
import numpy as np

class AerialInstanceLines:
    """Vector-instance wrapper compatible with SatMap's MapTRHead.

    Args:
        points: Sampled local-coordinate polylines, [N, fixed_num, 2].
        fixed_num: Number of points per polyline.
        padding_value: Sentinel used for invalid shift candidates.
        eps: Tolerance for detecting closed polylines.
    """

    def __init__(
        self,
        points,
        fixed_num=20,
        padding_value=-10000.0,
        eps=1e-5,
    ):
        points = np.asarray(points, dtype=np.float32)

        if points.size == 0:
            points = np.empty((0, fixed_num, 2), dtype=np.float32)

        if points.ndim != 3 or points.shape[1:] != (fixed_num, 2):
            raise ValueError(
                "Expected points shaped [N, {}, 2], got {}".format(
                    fixed_num, points.shape
                )
            )

        self.points = torch.from_numpy(
            np.ascontiguousarray(points)
        ).float()

        self.fixed_num = int(fixed_num)
        self.padding_value = float(padding_value)
        self.eps = float(eps)

    def __len__(self):
        return self.points.shape[0]

    @property
    def bbox(self):
        """Return [xmin, ymin, xmax, ymax] for each polyline: [N, 4]."""
        if len(self) == 0:
            return self.points.new_empty((0, 4))

        minimum = self.points.amin(dim=1)
        maximum = self.points.amax(dim=1)

        return torch.cat([minimum, maximum], dim=-1)

    @property
    def fixed_num_sampled_points(self):
        """Sampled points in the local coordinate system: [N, 20, 2]."""
        return self.points

    @property
    def shift_fixed_num_sampled_points_v2(self):
        """Generate v2 shift candidates: [N, 19, 20, 2].

        Closed lines:
            Rotate the unique sampled vertices in forward cyclic order,
            append the new starting point, and retain 19 candidates.

        Open lines:
            Use the forward and reversed point sequences, then pad the
            remaining candidates with padding_value.

        This follows the MapTR v2 convention of fixed_num - 1 candidates.
        """
        num_shifts = self.fixed_num - 1

        if len(self) == 0:
            return self.points.new_empty(
                (0, num_shifts, self.fixed_num, 2)
            )

        all_shifts = []

        for line in self.points:
            is_closed = torch.allclose(
                line[0],
                line[-1],
                atol=self.eps,
                rtol=0.0,
            )

            candidates = []

            if is_closed:
                # The final point duplicates the first point. Rotate only
                # the unique vertices, then close each rotated sequence.
                unique_points = line[:-1]
                num_unique = unique_points.shape[0]

                for shift in range(min(num_unique, num_shifts)):
                    rotated = torch.roll(
                        unique_points,
                        shifts=shift,
                        dims=0,
                    )
                    candidate = torch.cat(
                        [rotated, rotated[:1]],
                        dim=0,
                    )
                    candidates.append(candidate)

            else:
                # Open polylines have two valid orientations.
                candidates.append(line)
                candidates.append(torch.flip(line, dims=[0]))

            # Ensure every instance has exactly fixed_num - 1 candidates.
            if len(candidates) > num_shifts:
                candidates = candidates[:num_shifts]

            while len(candidates) < num_shifts:
                candidates.append(
                    line.new_full(
                        (self.fixed_num, 2),
                        self.padding_value,
                    )
                )

            all_shifts.append(torch.stack(candidates, dim=0))

        return torch.stack(all_shifts, dim=0)
