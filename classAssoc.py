'''
    Ties similar classes between the two
    datasets together.

    2020 Benjamin Kellenberger
'''

raw = [
    ['agricultural', 'farmland'],
    ['airplane', 'airport'],
    ['beach'],
    ['buildings','denseresidential', 'commercial', 'industrial'],
    ['forest'],
    ['harbor', 'port'],
    ['mediumresidential', 'residential'],
    ['overpass', 'viaduct'],
    ['parkinglot', 'parking'],
    ['river']
]

classAssoc = {}
classAssoc_inv = {}
for idx, classList in enumerate(raw):
    for className in classList:
        classAssoc[className] = idx
        classAssoc_inv[idx] = className