from xml.etree import ElementTree as ET

driver_list = [
    #"berniw",
    #"berniw2",
    #"berniw3",
    #"bt",
    "damned",
    #"inferno",
    #"inferno2",
    #"lliaw",
    #"olethros",
    #"tita",
    #"usr_2016",
]


track_list = [
    "street-1",
    "alpine-1",
    "e-track-1",
    "wheel-1",
    "ruudskogen",
    "spring",
    "g-track-2",
]


def change_track_name(node, driver):
    """Change driver module inplace"""
    module_node = node.find("./attstr[@name='name']")
    module_node.set("val", driver)

def change_driver(node, driver):
    """Change driver module inplace"""
    module_node = node.find('./attstr')
    module_node.set("val", driver)

def change_ai(name, driver):
    """Set AI and capture modules inplace"""

    tree = ET.parse(open(name))
    config = tree.getroot()
    drivers = config.find(".//section[@name='Drivers']")
    ai_driver = drivers.find("./section[@name='1']")
    capture_driver = drivers.find("./section[@name='2']")
    assert ai_driver is not None
    assert capture_driver is not None

    change_driver(ai_driver, driver)
    change_driver(capture_driver, "scr_server")
    tree.write(open(name, 'wb'))

def change_track(name, track_name):
    """Set track to drive on"""
    tree = ET.parse(open(name))
    config = tree.getroot()
    tracks = config.find(".//section[@name='Tracks']")
    track = tracks.find("./section[@name='1']")
    assert track is not None

    change_track_name(track, track_name)
    tree.write(open(name, 'wb'))
