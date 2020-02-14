'''
    Ties similar classes between the two
    datasets together.

    2020 Benjamin Kellenberger
'''

raw = [
    ['AG', 'agricultural', 'farmland'],
    ['AP', 'airplane', 'airport'],
    ['BE', 'beach'],
    ['DR', 'buildings','denseresidential', 'commercial', 'industrial'],
    ['FO', 'forest'],
    ['HA', 'harbor', 'port'],
    ['MR', 'mediumresidential', 'residential'],
    ['VI', 'overpass', 'viaduct'],
    ['PA', 'parkinglot', 'parking'],
    ['RI', 'river']
]

classAssoc = {}
classAssoc_inv = {}
for idx, classList in enumerate(raw):
    for className in classList:
        classAssoc[className] = idx
    classAssoc_inv[idx] = raw[idx][0]