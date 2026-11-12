"""rules.py -- lexicons and caption helpers of the selection rules (imported by flags.py and select.py).
All rules operate on public metadata only (TRELLIS-500K captions, Sketchfab counts from the Sketchfab table,
Objaverse 1.0 name/tags/categories/animationCount/user)."""
import re

TOK = re.compile(r"[a-z0-9]+")

# ---------------------------------------------------------------- lexicons
SCENE_HEAD = set("""scene scenes scenery diorama dioramas landscape landscapes terrain terrains island islands islet village villages
town towns city cities cityscape townscape street streets alley alleyway road roads crossroad crossroads intersection neighborhood
neighbourhood room rooms bedroom bedrooms kitchen kitchens bathroom bathrooms livingroom interior interiors office offices classroom
hallway corridor lobby courtyard yard backyard garden gardens park playground forest forests woods jungle field fields farm farmyard
ranch campsite camp battlefield ruins arena stadium market marketplace harbor harbour port dock docks pier marina beach shore coast
coastline cliff cliffs cave caves canyon valley mountain mountains hill hills hillside desert swamp lake pond river waterfall ocean
seabed reef environment environments area zone level levels map world worlds apartment apartments gallery museum library shop
shops storefront restaurant cafe diner pub tavern inn hotel factory warehouse hangar airport spaceport outpost compound settlement
colony cemetery graveyard dungeon labyrinth maze landscaping plaza square quarter district suburb slum junkyard scrapyard
campground trail path pathway walkway patio terrace deck balcony porch rooftop basement attic cellar studio showroom exhibition
panorama skyline skybox hdri background backdrop""".split())
# 'set' as head noun ("Chess set.", "Furniture set.") is a multi-object set, grouped here for the head test.
BUILDING_HEAD = set("""house houses home homes cabin cabins cottage cottages hut huts shack shacks villa villas mansion mansions building
buildings skyscraper skyscrapers tower towers castle castles palace palaces church churches chapel chapels cathedral cathedrals mosque
mosques temple temples shrine shrines pagoda pagodas monastery abbey lighthouse lighthouses windmill windmills barn barns farmhouse
bungalow bungalows condo duplex residence residences edifice facade facades pavilion shed sheds garage garages bridge bridges fort
fortress fortresses stronghold bunker bunkers mill station stations school hospital theater theatre cinema mausoleum tomb
citadel ziggurat stupa gatehouse greenhouse storehouse boathouse treehouse structure
structures skyscraper tavern highrise rise archway arch arches spire colonnade portico gateway minaret aqueduct dam""".split())
FLAT_HEAD = set("""poster posters painting paintings picture pictures photo photos photograph image images canvas card cards sticker
stickers decal decals logo logos text letter letters word words banner banners rug rugs carpet carpets mat mats doormat tapestry
wallpaper quad paper papers page pages document documents texture textures pattern patterns tile tiles floor
floors ground wall walls relief reliefs plaque plaques screenshot billboard billboards postcard label labels sign
signage artwork mural murals fresco frescoes mosaic inscription drawing drawings sketch sketches illustration
silhouette cutout emblem""".split())
# multiplicity / collection words used as a head or leading quantifier
MULTI_HEAD = set("""duo trio quartet pair pairs couple collection collections pack packs kit kits bundle bundles assortment assortments variety group groups pile piles
stack stacks array arrays lineup series bunch set sets pieces parts props assets items objects things models elements tools
weapons vehicles furniture accessories supplies decorations ornaments""".split())
QUANT = set("""two three four five six seven eight nine ten eleven twelve dozen several multiple many various assorted numerous
different""".split())
CONTEXT_NOUN = set("""ground grass grassy terrain hill hills field landscape scenery background room forest street road path
beach garden yard island dirt mud floor lawn meadow swamp cliff rubble rocks""".split())  # after on/in/among/atop/amid/inside
SURROUND_NOUN = set("""trees grass terrain landscape scenery background forest street road path garden yard fence fences bushes shrubs
vegetation lawn pool houses buildings""".split())  # after with/and/near/beside
SCENE_PHRASES = ["diorama", "landscape", "terrain", "cityscape", "floating island", "game level", "low poly scene", "lowpoly scene",
                 "multiple objects", "various objects", "assorted objects", "collection of", "set of various", "group of",
                 "assortment of", "variety of", "various poses", "different poses",
                 "multiple poses", "turnaround", "variations of",
                 "different versions", "various versions", "several versions", "various stages", "different stages",
                 "point cloud", "pointcloud", "interior view", "room interior", "scene with", "scene featuring",
                 "environment with", "an environment", "a scene", "the scene", "in a room", "on the ground", "on grass",
                 "on a grassy", "on grassy", "on terrain", "on a hill", "on rocky", "with trees", "with a tree", "with fence",
                 "with a fence", "with road", "with a road", "with a pool", "with pool", "with garden",
                 "with a garden", "with a yard", "with yard", "with background", "and trees", "and a tree", "and water",
                 "and grass", "and rocks", "and fence"]
NAME_TAG_SCENE = set("""diorama dioramas environment environments landscape terrain room level map city town
village island islands cityscape street neighborhood battlefield world worlds archviz""".split())
NAME_TAG_PACK = set("""pack packs bundle collection assetpack megapack kitbash modular tileset props assets""".split())
TAG_SCENE_STRONG = set("""diorama dioramas landscape terrain cityscape floatingisland floating-island gamelevel leveldesign level-design
archviz scene""".split())
TAG_PACK_STRONG = set("""assetpack asset-pack props-pack propspack megapack kitbash kit-bash modular tileset collection bundle""".split())
POINTCLOUD_WORDS = {"pointcloud", "point-cloud", "potree", "lidar", "splat", "splats", "gaussian-splatting", "gaussiansplatting",
                    "3dgs"}
PHONE_SCAN = {"polycam", "scaniverse", "kiri", "kiriengine", "kiri-engine", "trnio", "qlone", "3dscannerapp", "3d-scanner-app",
              "lumaai", "luma", "widar", "heges", "emesent", "iphone", "ipad", "ipadpro", "iphone12pro", "iphone13pro", "record3d",
              "canvas-io", "display-land", "displayland"}
SCAN_TAGS = {"photogrammetry", "3dscan", "3d-scan", "scan", "scanned", "scans", "realitycapture", "reality-capture", "agisoft",
             "metashape", "photoscan", "meshroom", "3dscanning", "3d-scanning", "scanning", "3df-zephyr", "zephyr", "artec",
             "structure-sensor", "sfm"} | PHONE_SCAN
WORLD_EXPORT = {"mineways", "jmc2obj", "worldpainter", "minecraft-world", "minecraftworld", "minecraft-map", "minecraftmap"}
ADJ = set("""black white red green blue yellow orange purple pink brown gray grey gold golden silver beige teal cyan magenta tan cream
dark light bright pale wooden wood metal metallic stone plastic leather old new big small large tiny round square tall short long
simple modern vintage rustic cute colorful colourful multicolored striped spotted shiny matte rusty worn navy maroon turquoise ivory
bronze copper crimson olive lime lavender violet peach mint khaki charcoal indigo scarlet emerald amber ruby off""".split())
SING_S = set("""pants jeans shorts trousers glasses sunglasses eyeglasses goggles binoculars scissors pliers tongs tweezers headphones
earphones earbuds clothes overalls news series species chassis canvas atlas gas lens dais bus octopus platypus cactus asparagus
hibiscus iris lotus compass harness mattress fortress press cross boss moss chess bass class circus abacus citrus walrus hippopotamus
rhinoceros tennis oasis axis crisis thesis apparatus nautilus uterus fungus hummus torus discus pegasus narcissus bonus campus
ammonites pancreas diabetes mathematics physics olympics gymnastics lens bellows gallows means pajamas pyjamas tights leggings
suspenders briefs panties crossbones dominos checkers cards darts""".split())


def toks(s):
    return TOK.findall(s.lower())


def is_plural(w):
    if w in SING_S or len(w) <= 3:
        return False
    if w.endswith(("ss", "us", "is", "ous", "ics", "ies") ) and not w.endswith("ies"):
        return False
    return w.endswith("s")


_ADJ_AND = re.compile(r"\b([a-z-]+) and ([a-z-]+)\b")
_SPLIT = re.compile(r",|;|:|\(|\s-\s|\s–\s|\bwith\b|\bon\b|\bin\b|\bfeaturing\b|\bfor\b|\bfrom\b|\batop\b|\bunder\b|\bover\b|"
                    r"\bnear\b|\bbeside\b|\bnext to\b|\bholding\b|\bwearing\b|\bsurrounded by\b|\binside\b|\bat\b|\bby\b|"
                    r"\bshowing\b|\bof\b|\bagainst\b|\bwhich\b|\bthat\b|\bdepicting\b|\bresembling\b|\bshaped\b")


def head_analysis(caption):
    """Return (head_noun, first_word, noun_and_conjunction, n_commas) of a (short) caption."""
    c = caption.lower().strip().rstrip(".!")
    c = re.sub(r"\"[^\"]*\"|'[^']*'|“[^”]*”", " ", c)
    n_commas = c.count(",")
    # adjective conjunctions ("black and white vase") are not noun lists
    def _adj(m):
        a, b = m.group(1), m.group(2)
        return f"{a}-{b}" if (a in ADJ or a.endswith("-colored") or a.endswith("ed")) else m.group(0)
    c2 = _ADJ_AND.sub(_adj, c)
    seg = _SPLIT.split(c2, maxsplit=1)[0]
    noun_and = False
    if " and " in seg:
        noun_and = True
        seg = seg.split(" and ")[0]
    words = TOK.findall(seg.replace("-", " "))
    words = [w for w in words if w not in ("a", "an", "the", "3d", "model", "low", "poly", "lowpoly", "stylized", "cartoon")] or TOK.findall(seg)
    head = words[-1] if words else ""
    first = TOK.findall(c)[0] if TOK.findall(c) else ""
    return head, first, noun_and, n_commas
