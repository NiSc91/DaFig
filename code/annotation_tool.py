from config import *

# Variables
handler = CollectionHandler(CORPORA_DIR)
all_corpora = handler.get_collections()
print("All annotated corpora:", all_corpora)

# Create paths for the agreement corpora
CORPUS_NAMES = ['main', 'agr1', 'agr2', 'agr3', 'agr_combined', 'consensus']
CORPUS_PATHS = {f"{name.upper()}_PATH": handler.get_collection_path(os.path.join(CORPORA_DIR, name)) for name in CORPUS_NAMES}
OUTPUT_DIR = os.path.join(CORPORA_DIR, "../BIO")

# Check if the output directory exists, if not, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Create a lambda function to get ann paths
get_ann_path = lambda base_path, ann_folder: os.path.join(base_path, ann_folder)

# Create a dictionary with the paths to the ann folders for each corpus except for main
base_paths = {name: handler.get_collection_path(os.path.join(CORPORA_DIR, name)) for name in CORPUS_NAMES[1:]}
ann_paths = {f"{name.upper()}_ANN1_PATH": get_ann_path(base_path, 'ann1') for name, base_path in base_paths.items()}
ann_paths.update({f"{name.upper()}_ANN2_PATH": get_ann_path(base_path, 'ann2') for name, base_path in base_paths.items()})

class AnnotationTool:
    def __init__(self):
        self.text = ""
        self.annotations = {
            'T': [],  # Text spans
            'A': [],  # Attributes
            'R': [],  # Relations
            '#': []   # Notes
        }
        self.current_t_index = 1
        self.current_a_index = 1
        self.current_r_index = 1
        self.current_note_index = 1

    def load_text(self, filename):
        """Load the source text file"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                self.text = file.read()
            print(f"Loaded text file: {len(self.text)} characters")
        except FileNotFoundError:
            print("File not found. Please check the filename.")

    def add_text_span(self):
        """Add a text span annotation (T)"""
        print("\nCurrent text:", self.text)
        start = int(input("Enter start offset: "))
        end = int(input("Enter end offset: "))
        print(f"Selected text: '{self.text[start:end]}'")
        
        annotation_type = input("Enter annotation type (e.g., MTP, HPB): ")
        
        # Create T annotation
        t_annotation = f"T{self.current_t_index} {annotation_type} {start} {end} {self.text[start:end]}"
        self.annotations['T'].append(t_annotation)
        
        # Add attributes
        while True:
            add_attribute = input("Add attribute? (y/n): ").lower()
            if add_attribute != 'y':
                break
                
            attr_name = input("Enter attribute name (e.g., Directness, Conventionality): ")
            attr_value = input("Enter attribute value: ")
            a_annotation = f"A{self.current_a_index} {attr_name} T{self.current_t_index} {attr_value}"
            self.annotations['A'].append(a_annotation)
            self.current_a_index += 1
        
        # Add note if needed
        add_note = input("Add note? (y/n): ").lower()
        if add_note == 'y':
            note = input("Enter note text: ")
            note_annotation = f"#{self.current_note_index} AnnotatorNotes T{self.current_t_index} {note}"
            self.annotations['#'].append(note_annotation)
            self.current_note_index += 1
        
        self.current_t_index += 1

    def add_relation(self):
        """Add a relation annotation (R)"""
        rel_type = input("Enter relation type (e.g., MultiWordExpression): ")
        arg1 = input("Enter Arg1 (T number): ")
        arg2 = input("Enter Arg2 (T number): ")
        
        r_annotation = f"R{self.current_r_index} {rel_type} Arg1:{arg1} Arg2:{arg2}"
        self.annotations['R'].append(r_annotation)
        self.current_r_index += 1

    def save_annotations(self, filename):
        """Save annotations to .ann file"""
        with open(filename, 'w', encoding='utf-8') as f:
            # Write T annotations
            for t in self.annotations['T']:
                f.write(t + '\n')
            
            # Write A annotations
            for a in self.annotations['A']:
                f.write(a + '\n')
            
            # Write R annotations
            for r in self.annotations['R']:
                f.write(r + '\n')
            
            # Write # annotations
            for n in self.annotations['#']:
                f.write(n + '\n')
        
        print(f"Annotations saved to {filename}")

    def run(self):
        """Main interaction loop"""
        print("\nCommands:")
        print("T: Add text span annotation")
        print("R: Add relation")
        print("S: Save annotations")
        print("Q: Quit")
    
        while True:
            command = input("\nEnter command: ").upper()
    
            if command == 'T':
                self.add_text_span()
            elif command == 'R':
                self.add_relation()
            elif command == 'S':
                output_file = os.path.splitext(self.text_file)[0] + ".ann"
                self.save_annotations(output_file)
            elif command == 'Q':
                if input("Save before quitting? (y/n): ").lower() == 'y':
                    output_file = os.path.splitext(self.text_file)[0] + ".ann"
                    self.save_annotations(output_file)
                break

def main():
    tool = AnnotationTool()
    collection_name = input("Enter the name of the collection: ")
    collection_path = handler.get_collection_path(os.path.join(CORPORA_DIR, collection_name))

    if collection_path:
        txt_files = handler.get_txt_files(collection_path)
        for filename in txt_files:
            text_file_path = os.path.join(collection_path, filename)
            tool.load_text(text_file_path)
            tool.run()
    else:
        print(f"Collection '{collection_name}' not found.")

if __name__ == "__main__":
    main()
