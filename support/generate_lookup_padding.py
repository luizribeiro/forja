"""Reproduce the original, deterministic background documents; no external corpus."""
from pathlib import Path
import random

SEED = 20261109
DOMAINS = [
    ("Coastal survey", "tidal", "cove shoal estuary channel inlet lagoon dune reef"),
    ("Meteor catalogue", "orbital", "comet meteor radiant ellipse aphelion perihelion transit occultation"),
    ("Botanical transect", "seasonal", "lichen moss fern sedge willow birch alder heather"),
    ("Ceramic archive", "mineral", "ochre feldspar kaolin quartz mica cobalt silica basalt"),
    ("Bird census", "migratory", "tern curlew plover gull egret heron swift swallow"),
    ("Acoustic survey", "resonant", "harmonic overtone cadence interval spectrum timbre echo pulse"),
    ("Glacier traverse", "alpine", "moraine crevasse cirque ridge saddle basin scree summit"),
    ("Textile collection", "woven", "linen hemp wool silk cotton jute ramie flax"),
    ("Fossil inventory", "stratigraphic", "ammonite trilobite crinoid coral brachiopod nautilus echinoid bryozoan"),
    ("Orchard census", "pomological", "quince pear plum medlar apricot mulberry cherry fig"),
    ("Ocean station", "pelagic", "salinity thermocline plankton current eddy upwelling turbidity oxygen"),
    ("Volcanic notes", "igneous", "pumice andesite obsidian scoria tuff rhyolite lava ash"),
    ("Pollen atlas", "palynological", "exine aperture grain tetrad reticulum pore annulus colpus"),
    ("Cloud log", "atmospheric", "cirrus stratus cumulus nimbus virga halo sundog contrail"),
    ("Insect census", "entomological", "moth beetle cicada damselfly lacewing firefly mayfly mantis"),
    ("Dye archive", "chromatic", "madder indigo weld woad saffron logwood cochineal annatto"),
]
FORMS = [
    "At station {n}, {a} occupied {x} percent of the marked area; {b} lay beyond the eastern boundary.",
    "The {adjective} register assigns specimen {n} to {a}, with a separate cross-reference for {b}.",
    "Observation {n} describes {a} at depth {x} centimeters and records no comparable {b} in the adjacent sample.",
    "A pencil annotation beside entry {n} questions whether {a} and {b} belong to the same seasonal sequence.",
    "Index {n} retains the older name {a}; the later margin note uses {b} and gives a count of {x}.",
    "Plate {n} places {a} above {b}; the scale bar represents {x} millimeters under diffuse illumination.",
    "For sample {n}, the observer classified {a} as abundant and {b} as sparse across {x} subdivisions.",
    "Section {n} compares the orientation of {a} with {b}, using an azimuth of {x} degrees from north.",
    "Ledger mark {n} denotes {a}; a blue underline distinguishes the associated {b} from the neighboring entries.",
    "The field sketch numbered {n} shows {a} on the western side and {b} near the outermost contour.",
    "Catalogue slip {n} gives {a} a provisional date and associates {b} with the interval ending at day {x}.",
    "Under reference {n}, {a} appeared before {b}; the accompanying diagram preserves {x} separate marks.",
]


def render():
    rng = random.Random(SEED)
    documents = []
    for number in range(256):
        title, adjective, vocabulary = DOMAINS[number % len(DOMAINS)]
        words = vocabulary.split()
        sentences = []
        for item, form in enumerate(rng.sample(FORMS, len(FORMS))):
            a, b = rng.sample(words, 2)
            sentences.append(form.format(n=number * 100 + item + 1, a=a, b=b,
                                         x=rng.randrange(3, 347), adjective=adjective))
        documents.append(f"Document {number + 1:03d}: {title}\n" + " ".join(sentences))
    return "\n\n".join(documents) + "\n"


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "bench/prompts/lookup/context.txt"
    destination.write_text(render())
