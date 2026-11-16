# Known issues in earlier releases

## Repeated acquisitions packed into one volume (0.9.5rc2 and earlier)

Releases up to and including 0.9.5rc2 (the last final release affected is
0.9.4; tested from 0.7.0) can return a "volume" that holds two or more
acquisitions of the same slices. Results computed from such a volume, such
as an ADC map fitted per volume, a classification, or a converted NIfTI
file, can be wrong without any error being raised. Release 0.9.6 and later
split these acquisitions; current releases also handle the uneven cases
described below.

### Why it happens

Those releases form volumes only from ITK's `GDCMSeriesFileNames` groups.
Files share a group when they share the SeriesInstanceUID, GDCM's series
details (series number, sequence name, slice thickness, rows, and
columns), and the values of the diffusion tags dcm-classifier adds as
restrictions (b-value, gradient direction, B-matrix, and echo number,
standard and vendor-private). A series is packed when it contains several
acquisitions of the same slice positions that differ only in tags outside
that list: AcquisitionNumber, AcquisitionTime, TemporalPositionIdentifier,
or TriggerTime.

### Series it affects

| Acquisition                                                | Returned by 0.9.4                                     |
| ---------------------------------------------------------- | ----------------------------------------------------- |
| Diffusion with a repeated direction, or several b=0 images | one volume per repeated encoding, holding all repeats |
| Perfusion, fMRI, or any dynamic series                     | one stack of every timepoint                          |
| Siemens mosaic diffusion with repeats                      | one volume of several mosaic files                    |

Measured on Siemens clinical series with 0.9.1 and 1.1.0:

| Series                                                   | Files | Positions | 0.9.1            | 1.1.0            |
| -------------------------------------------------------- | ----- | --------- | ---------------- | ---------------- |
| Raw DWI, 26 acquisitions in 13 identically encoded pairs | 624   | 24        | 13 volumes of 48 | 26 volumes of 24 |
| Perfusion, 60 timepoints                                 | 1140  | 19        | 1 volume of 1140 | 60 volumes of 19 |
| Mosaic DWI, 26 acquisitions                              | 26    | 1         | 13 volumes of 2  | 26 volumes of 1  |

Within each series the acquisitions differ only in AcquisitionNumber.

### Checking a volume

A packed volume repeats ImagePositionPatient values. With any release:

```python
from collections import Counter
import pydicom

positions = Counter(
    tuple(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient)
    for f in volume.get_one_volume_dcm_filenames()
)
assert max(positions.values()) == 1, "volume holds repeated acquisitions"
```

In current releases, `volume.get_slice_ordering()` reports how each volume
was formed, and a study with any volume that is not a position-ordered 3D image
raises `ValueError` unless the caller passes `allow_fallback_ordering=True`
(`--allow_fallback_ordering` on the command line).

## Uneven repeats dropped or packed (0.9.6 to 1.1.0)

Releases 0.9.6 to 1.1.0 split repeated acquisitions by counting how often
each slice position repeats, then taking the k-th repeat at every position
as volume k. When the counts differ between positions, the result depends on
the counts, with only a log warning:

| Series                                                     | Files | 1.1.0                         | Current    |
| ---------------------------------------------------------- | ----- | ----------------------------- | ---------- |
| 4 timepoints of 6 slices, the last stopped after 3         | 21    | 6, 6, 6 (3 files dropped)     | 6, 6, 6, 3 |
| 2 acquisitions of 6 slices, the second shifted one slice   | 12    | 7 (slices of both, 5 dropped) | 6, 6       |
| 3 acquisitions of 6 slices, the third shifted three slices | 18    | 18 (all packed)               | 6, 6, 6    |

Current releases split by TemporalPositionIdentifier or AcquisitionNumber
when those tags separate the repeats, and both must agree when both do.
Otherwise they split by repeat order only when InstanceNumber separates the
repeats and every acquisition time that parses agrees with that order.
Where a second header can check the first, a disagreement refuses the
series; a wrong value in the only header that separates the repeats cannot
be detected. They keep every file, and label a series they cannot separate
`fallback_repeated_positions`, which is refused by default.
Series numbered slice by slice, or restarting at each timepoint, with
neither tag to tell the repeats apart are refused for that reason, as are
series whose two tags both separate the repeats but group them
differently, such as one per timepoint and the other per slab.
