import xml.etree.ElementTree as ET

def load_annotations(annotation_file):
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    return root  # You can process the XML data as needed

# Call the function with the correct XML file
annotation = load_annotations("jaad_dataset/JAAD-JAAD_2.0/annotations/video_0001.xml")

# Example: Print XML structure
for child in annotation:
    print(child.tag, child.attrib)
