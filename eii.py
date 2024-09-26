# Panos Lelakis, Eternity II

# Import libraries
import os
import pulp
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# GUI class
class EternityIISolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Eternity II Solver")
        self.w = 100
        self.h = self.w
        
        # Variables to store file paths and checkbox state
        self.pieces_file_path = None
        self.hint_file_path = None
        self.time_limit = None
        self.solution_found = False  # Track if a solution has been found

        # Create GUI elements
        self.create_widgets()
        self.solution_board = []
        self.solution_text = ""  # To store the solution text for display

        # Empty canvas for the solution board
        canvas = tk.Canvas(self.board_frame, width=500, height=500, borderwidth=0, highlightthickness=0)
        canvas.grid(row=1, column=1, padx=0, pady=0)

    # Create widgets in the window
    def create_widgets(self):
        # Left-side frame for inputs and buttons
        left_frame = tk.Frame(self.root)
        left_frame.grid(row=0, column=0, sticky='n')

        # HELP button
        self.help_btn = tk.Button(left_frame, text="HELP", command=self.show_help)
        self.help_btn.pack(pady=10)

        # Pieces file selection
        tk.Label(left_frame, text="Select Pieces Data:").pack(pady=10)
        self.pieces_file_btn = tk.Button(left_frame, text="Browse", command=self.select_pieces_file)
        self.pieces_file_btn.pack()

        # Checkbox for pieces file selection
        self.pieces_file_checkbox = tk.Checkbutton(left_frame, text="Pieces File Selected", state=tk.DISABLED)
        self.pieces_file_checkbox.pack()

        # Hints file selection
        tk.Label(left_frame, text="Select Hint Pieces (Optional):").pack(pady=10)
        self.hint_file_btn = tk.Button(left_frame, text="Browse", command=self.select_hint_file)
        self.hint_file_btn.pack()

        # Checkbox for hint pieces file selection
        self.hint_file_checkbox = tk.Checkbutton(left_frame, text="Hint Pieces File Selected", state=tk.DISABLED)
        self.hint_file_checkbox.pack()

        # Time limit entry
        tk.Label(left_frame, text="Solver Time Limit (seconds):").pack(pady=5)
        self.time_limit_entry = tk.Entry(left_frame)
        self.time_limit_entry.pack()

        # Calculate button
        self.calculate_btn = tk.Button(left_frame, text="Calculate", command=self.calculate_solution)
        self.calculate_btn.pack(pady=20)

        # Reset button
        self.reset_btn = tk.Button(left_frame, text="Reset", command=self.reset)
        self.reset_btn.pack(pady=20)

        # View positions and rotations button (Initially disabled)
        self.view_solution_btn = tk.Button(left_frame, text="View positions and rotations", command=self.view_solution, state=tk.DISABLED)
        self.view_solution_btn.pack(pady=10)

        # Creator hyperlink
        creator_label = tk.Label(left_frame, text="Creator", fg="blue", cursor="hand2")
        creator_label.pack(pady=5)
        # Source for hyperlink label:
        # https://stackoverflow.com/questions/23482748/how-to-create-a-hyperlink-with-a-label-in-tkinter
        creator_label.bind("<Button-1>", lambda e: self.open_link("https://github.com/PanosLelakis"))

        # Solution board frame (right side)
        self.board_frame = tk.Frame(self.root)
        self.board_frame.grid(row=0, column=1, padx=20, pady=20)

    # Show help window
    def show_help(self):
        # Create a new window
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")

        # Create a read-only text field
        text_field = tk.Text(help_window, height=45, width=100, wrap=tk.WORD)
        text_field.pack()

        # Read the content from help.txt file and insert it into the text field
        try:
            with open("help.txt", "r") as help_file:
                help_content = help_file.read()
                text_field.insert(tk.END, help_content)
        except Exception as e:
            text_field.insert(tk.END, "Error reading help file: " + str(e))

        # Make the text field read-only
        text_field.config(state=tk.DISABLED)

    # Load pieces information from txt file
    def select_pieces_file(self):
        self.pieces_file_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Pieces Data", filetypes=[("Text Files", "*.txt")])
        if self.pieces_file_path:
            self.pieces_file_checkbox.select()

    # Load hint pieces information from txt file
    def select_hint_file(self):
        self.hint_file_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Hint Pieces", filetypes=[("Text Files", "*.txt")])
        if self.hint_file_path:
            self.hint_file_checkbox.select()

    # Open hyperlink
    def open_link(self, url):
        import webbrowser
        webbrowser.open_new(url)

    # Reset solver
    def reset(self):
        # Clear the selections
        self.pieces_file_path = None
        self.hint_file_path = None
        self.time_limit = None
        self.solution_found = False
        self.pieces_file_checkbox.deselect()
        self.hint_file_checkbox.deselect()
        # Clear the previous solution display
        for widget in self.board_frame.winfo_children():
            widget.destroy()
        self.solution_board.clear() # Clear solution board
        self.time_limit_entry.delete(0, tk.END) # Clear the time limit entry field
        self.view_solution_btn.config(state=tk.DISABLED)
        print("Reset completed.")

    # Model the problem and solve it
    def calculate_solution(self):
        # If there are no info about pieces, throw error
        if not self.pieces_file_path:
            messagebox.showerror("Error", "Please select a pieces data file.")
            return
        
        # Get time limit from entry field
        try:
            self.time_limit = int(self.time_limit_entry.get()) if self.time_limit_entry.get() else None
        except ValueError:
            messagebox.showerror("Error", "Time limit must be a number.")
            return

        n, m, pieces_data, L = self.load_pieces_and_size(self.pieces_file_path) # Load pieces

        # Load hint pieces if given
        if self.hint_file_path: 
            hint_pieces = self.load_hint_pieces(self.hint_file_path)
        else:
            hint_pieces = []

        print(f"Puzzle size: {n} x {m}")
        print(f'Unique combinations: {L}')

        # Solve puzzle
        prob, solving_time = self.solve_puzzle(n, m, pieces_data, hint_pieces, L, self.time_limit) # Solve the puzzle

        # If solution is found, proceed
        if prob:
            solution = self.extract_solution(prob) # Extract solution (simplify format)
            self.print_solution(solution) # Print solution
            self.display_solution(n, m, solution, pieces_data, L) # Display solution on the solution board
            shared_edges, bad_edges = self.calculate_edges(solution, pieces_data, n, m) # Calculate shared and bad edges
            result_filename = f"results/results_{n}x{m}.txt"
            self.save_result_to_file(solution, n, m, shared_edges, bad_edges, solving_time, result_filename) # Save results to txt file
            self.solution_text = self.get_solution_text(solution)  # Get the formatted solution text
            self.solution_found = True
            self.view_solution_btn.config(state=tk.NORMAL)  # Enable the "View positions and rotations" button
            print(f"Results saved to {result_filename}")
        
        # If no solution is found, throw error
        else:
            messagebox.showerror("Error", "Failed to calculate the solution.")
            return

    def get_solution_text(self, solution):
        solution_str = ""
        for (t, r, c, a), value in solution.items():
            if value == 1:
                solution_str += f"Piece {t}: row {r}, column {c} with rotation {a}\n"
        return solution_str
    
    # View positions and rotations (opens a new window)
    def view_solution(self):
        if self.solution_found:
            # Create a new window
            solution_window = tk.Toplevel(self.root)
            solution_window.title("Solution: Positions and Rotations")

            # Create a read-only text field
            text_field = tk.Text(solution_window, height=20, width=60, wrap=tk.WORD)
            text_field.insert(tk.END, self.solution_text)
            text_field.config(state=tk.DISABLED)  # Make the text field read-only
            text_field.pack()

    # Load pieces info and board size
    def load_pieces_and_size(self, filename):
        with open(filename, 'r') as file:
            first_line = file.readline().strip() # Read the first line to get board dimensions
            n, m = map(int, first_line.split())  # Save board dimensions
            pieces = np.loadtxt(file, dtype=int)  # Load pieces as 2D array
        unique_combinations = self.calculate_combinations(pieces) # Get unique color combinations
        return n, m, pieces, sorted(unique_combinations)

    # Load hint pieces info
    def load_hint_pieces(self, filename):
        hint_pieces = []
        with open(filename, 'r') as file:
            for line in file:
                hint_pieces.append(tuple(map(int, line.strip().split())))
        return hint_pieces

    # Find all unique color combinations
    def calculate_combinations(self, pieces_data):
        unique_combinations = set()
        for piece in pieces_data:
            for combination in piece:
                unique_combinations.add(combination)
        return unique_combinations
    
    # Get the side combinations of a given piece (left, top, right, bottom)
    def get_sides(self, piece, rotation, pieces_data):
        left, top, right, bottom = pieces_data[piece]
        if rotation == 1:
            return [bottom, left, top, right]
        elif rotation == 2:
            return [right, bottom, left, top]
        elif rotation == 3:
            return [top, right, bottom, left]
        else:
            return [left, top, right, bottom]
    
    # Define the CT, CB, CL, CR coefficients
    def CT(self, t, a, l, pieces_data):
        top_side = self.get_sides(t, a, pieces_data)[1]
        return 1 if top_side == l else 0

    def CB(self, t, a, l, pieces_data):
        bottom_side = self.get_sides(t, a, pieces_data)[3]
        return 1 if bottom_side == l else 0

    def CL(self, t, a, l, pieces_data):
        left_side = self.get_sides(t, a, pieces_data)[0]
        return 1 if left_side == l else 0

    def CR(self, t, a, l, pieces_data):
        right_side = self.get_sides(t, a, pieces_data)[2]
        return 1 if right_side == l else 0

    # Implement MILP model and solve it
    # Source for pulp docs:
    # https://www.coin-or.org/PuLP/pulp.html
    def solve_puzzle(self, n, m, pieces_data, hint_pieces, L, time_limit=None):
        # Initialize minimization problem
        prob = pulp.LpProblem("Eternity_II", pulp.LpMinimize)

        # Sources for adding t r c a (multiple data) in one variable:
        # https://stackoverflow.com/questions/49490423/formulating-an-linear-programming-assignment-in-pulp?rq=3
        # https://stackoverflow.com/questions/67238034/pulp-adding-multiple-lpsum-in-a-loop?rq=3

        # Add decision variable x for the tiles
        x = pulp.LpVariable.dicts("x", [(t, r, c, a) for t in range(n * m)
                                        for r in range(n)
                                        for c in range(m)
                                        for a in range(4)], cat="Binary")

        # Add horizontal (h) and vertical (v) decision variables for edge matching
        h = pulp.LpVariable.dicts("h", [(r, c) for r in range(n) for c in range(m - 1)], cat="Binary")
        v = pulp.LpVariable.dicts("v", [(r, c) for r in range(n - 1) for c in range(m)], cat="Binary")

        # Sources for adding sums in multiple constraints in loops:
        # https://stackoverflow.com/questions/51438018/pulp-objective-function-adding-multiple-lpsum-in-a-loop
        # https://stackoverflow.com/questions/30340431/formulating-constraints-of-a-lp-in-pulp-python?rq=3 


        # (1) Objective function: minimize unmatched edges
        prob += pulp.lpSum(h[r, c] for r in range(n) for c in range(m - 1)) + pulp.lpSum(v[r, c] for r in range(n - 1) for c in range(m)), "Objective"

        # (2) Constraint to ensure each piece is placed exactly once
        for t in range(n*m):
            prob += pulp.lpSum(x[t, r, c, a] for r in range(n) for c in range(m) for a in range(4)) == 1

        # (3) Constraint to ensure each cell has exactly one piece
        for r in range(n):
            for c in range(m):
                prob += pulp.lpSum(x[t, r, c, a] for t in range(n*m) for a in range(4)) == 1

        # (4) Horizontal edge matching constraint (color l matches between (r, c) and (r, c+1))
        for r in range(n):
            for c in range(m - 1):
                for l in L[1:]:  # L is the total number of combinations
                    prob += (pulp.lpSum(self.CR(t1, a1, l, pieces_data) * x[t1, r, c, a1]
                                        for t1 in range(n * m) for a1 in range(4))
                            - pulp.lpSum(self.CL(t2, a2, l, pieces_data) * x[t2, r, c+1, a2]
                                        for t2 in range(n * m) for a2 in range(4))
                            - h[r, c]) <= 0

        # (5) Symmetric counterpart of constraint (4)
        for r in range(n):
            for c in range(m - 1):
                for l in L[1:]:
                    prob += (- pulp.lpSum(self.CR(t1, a1, l, pieces_data) * x[t1, r, c, a1]
                                        for t1 in range(n * m) for a1 in range(4))
                            + pulp.lpSum(self.CL(t2, a2, l, pieces_data) * x[t2, r, c+1, a2]
                                        for t2 in range(n * m) for a2 in range(4))
                            - h[r, c]) <= 0

        # (6) Vertical edge matching constraint
        for r in range(n - 1):
            for c in range(m):
                for l in L[1:]:
                    prob += (pulp.lpSum(self.CB(t1, a1, l, pieces_data) * x[t1, r, c, a1]
                                        for t1 in range(n * m) for a1 in range(4))
                            - pulp.lpSum(self.CT(t2, a2, l, pieces_data) * x[t2, r+1, c, a2]
                                        for t2 in range(n * m) for a2 in range(4))
                            - v[r, c]) <= 0

        # (7) Symmetric counterpart of constraint (6)
        for r in range(n - 1):
            for c in range(m):
                for l in L[1:]:
                    prob += (- pulp.lpSum(self.CB(t1, a1, l, pieces_data) * x[t1, r, c, a1]
                                        for t1 in range(n * m) for a1 in range(4))
                            + pulp.lpSum(self.CT(t2, a2, l, pieces_data) * x[t2, r+1, c, a2]
                                        for t2 in range(n * m) for a2 in range(4))
                            - v[r, c]) <= 0

        # (8) Constraint to ensure top edge pieces have grey (combination 0) on top
        for c in range(m):
            prob += pulp.lpSum(self.CT(t, a, 0, pieces_data) * x[t, 0, c, a] for t in range(n * m) for a in range(4)) == 1

        # (9) Constraint to ensure bottom edge pieces have grey (combination 0) on bottom
        for c in range(m):
            prob += pulp.lpSum(self.CB(t, a, 0, pieces_data) * x[t, n-1, c, a] for t in range(n * m) for a in range(4)) == 1

        # (10) Constraint to ensure left edge pieces have grey (combination 0) on left
        for r in range(n):
            prob += pulp.lpSum(self.CL(t, a, 0, pieces_data) * x[t, r, 0, a] for t in range(n * m) for a in range(4)) == 1

        # (11) Constraint to ensure right edge pieces have grey (combination 0) on right
        for r in range(n):
            prob += pulp.lpSum(self.CR(t, a, 0, pieces_data) * x[t, r, m-1, a] for t in range(n * m) for a in range(4)) == 1

        # (15) Constraint for hint pieces
        for (th, rh, ch, ah) in hint_pieces:
            prob += x[th, rh, ch, ah] == 1

        # Solve the problem using CBC solver
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
        
        start_time = time.time() # Start timer
        prob.solve(solver)
        end_time = time.time() # End timer
        
        # Calculate time taken to solve
        solving_time = end_time - start_time
        print(f"Time to solve the puzzle: {solving_time:.2f} seconds")

        # After solving, calculate shared and bad edges, and percent matched
        if prob:
            solution = self.extract_solution(prob)
            shared_edges, bad_edges = self.calculate_edges(solution, pieces_data, n, m)
            total_edges = (n * (m - 1)) + ((n - 1) * m)  # Total internal edges
            percent_matched = round((shared_edges / total_edges) * 100)
            
            # Log the solution details
            self.log_solution(n, m, len(hint_pieces), time_limit, percent_matched, solving_time)
            return prob, solving_time
        else:
            return None, None

    # Extract the solution
    def extract_solution(self, prob):
        solution = {}
        for v in prob.variables():
            # Only process variables starting with "x_" (ignore h and v)
            if v.name.startswith("x_"):
                clean_name = v.name.replace("x_(", "").replace(")", "").replace("_", "")
                name_parts = clean_name.split(",")
                try:
                    t, r, c, a = map(int, name_parts)  # Convert to integers
                    if v.varValue == 1:  # If this variable is set to 1
                        solution[(t, r, c, a)] = 1
                except ValueError:
                    print(f"Error parsing variable name: {v.name}")
        return solution
    
    # Print the solution
    def print_solution(self, solution):
        for (t, r, c, a), value in solution.items():
            if value == 1:  # Assuming the value 1 indicates a part of the solution
                print(f"Piece {t}: row {r}, column {c} with rotation {a}")
    
    # Function to create a red line for non-matching edges
    def draw_red_line(self, canvas, start_x, start_y, end_x, end_y):
        canvas.create_line(start_x, start_y, end_x, end_y, fill='red', width=10)

    # Display solution in solution board
    def display_solution(self, n, m, solution, pieces_data, combinations):
        # Clear leftover widgets in solution board
        for widget in self.board_frame.winfo_children():
            widget.destroy()
        
        # Load images
        images = {}
        ratio = 1.4
        w = self.w
        h = self.h
        for i in combinations:
            try:
                img = Image.open(f"images_combinations/{i}.png") # Load the image
                img = img.resize((round(144/ratio), round(72/ratio)), Image.Resampling.LANCZOS) # Resize the image to fit the triangles
                images[i] = img # Save image
            except FileNotFoundError:
                print(f"Image for combination {i} not found.")

        # Define coordinates for the triangles
        points = [[0, 0, w / 2, h / 2, 0, h],  # Left triangle
            [0, 0, w / 2, h / 2, w, 0],  # Top triangle
            [w, 0, w / 2, h / 2, w, h],  # Right triangle
            [0, h, w / 2, h / 2, w, h]]  # Bottom triangle

        # Create a grid to hold the pieces
        for r in range(n):
            for c in range(m):
                piece_id = None
                for (t, row, col, a) in solution:
                    if row == r and col == c:
                        piece_id = t
                        break

                # Create a canvas for each cell
                canvas = tk.Canvas(self.board_frame, width=w, height=h, borderwidth=0, highlightthickness=0)
                canvas.grid(row=r, column=c, padx=0, pady=0)

                if piece_id is not None:
                    sides = self.get_sides(piece_id, a, pieces_data) # Get color of each side of the piece

                    # Draw triangles and fill with color combination numbers
                    for i in range(4):                    
                        canvas.create_polygon(points[i], fill="", outline="black")  # Draw the triangle

                        # Calculate the center of each triangle for text placement
                        if i == 0:  # Left triangle
                            text_x, text_y = (w/4, h/2)
                        elif i == 1:  # Top triangle
                            text_x, text_y = (w/2, h/4)
                        elif i == 2:  # Right triangle
                            text_x, text_y = (3*w/4, h/2)
                        elif i == 3:  # Bottom triangle
                            text_x, text_y = (w/2, 3*h/4)

                        # If color combination is known and there is an image for it, display the image
                        if sides[i] in images:
                            img = images[sides[i]]

                            # Rotate the image based on the triangle's orientation
                            if i == 0:  # Left triangle (rotate 90 degrees counterclockwise)
                                rotated_img = img.rotate(90, expand=True)
                                img_x, img_y = (w/4, h/2)
                            elif i == 2:  # Right triangle (rotate 90 degrees clockwise)
                                rotated_img = img.rotate(270, expand=True)
                                img_x, img_y = (3*w/4, h/2)
                            elif i == 3:  # Bottom triangle (rotate 180 degrees)
                                rotated_img = img.rotate(180, expand=True)
                                img_x, img_y = (w/2, 3*h/4)
                            else:  # Top triangle (no rotation)
                                rotated_img = img
                                img_x, img_y = (w/2, h/4)
                            
                            tk_img = ImageTk.PhotoImage(rotated_img) # Convert the rotated image to a Tkinter-compatible image
                            self.solution_board.append(tk_img) # Add image to solution board
                            canvas.create_image(img_x, img_y, image=tk_img) # Place the image inside the triangle
                        
                        # If color combination is unknown and there is no image for it, display the combination number
                        else:
                            canvas.create_text(text_x, text_y, text=str(sides[i]), font=("Arial", 10), anchor='center')

                # Red lines for unmatched edges
                if c < m - 1:  # Check right edge
                    right_piece_id = None
                    for (t, row, col, a) in solution:
                        if row == r and col == c + 1:
                            right_piece_id = t
                            break
                    if right_piece_id is not None:
                        right_sides = self.get_sides(right_piece_id, a, pieces_data)
                        if sides[2] != right_sides[0]:  # Mismatch on the right
                            self.draw_red_line(canvas, w, h, w, 0)

                if r < n - 1:  # Check bottom edge
                    bottom_piece_id = None
                    for (t, row, col, a) in solution:
                        if row == r + 1 and col == c:
                            bottom_piece_id = t
                            break
                    if bottom_piece_id is not None:
                        bottom_sides = self.get_sides(bottom_piece_id, a, pieces_data)
                        if sides[3] != bottom_sides[1]:  # Mismatch on the bottom
                            self.draw_red_line(canvas, w, h, 0, h)

    # Calculate shared and bad edges based on the solution
    def calculate_edges(self, solution, pieces_data, n, m):
        shared_edges = 0
        bad_edges = 0

        # Create a board to store piece IDs and rotations
        board = np.zeros((n, m, 2), dtype=int)  # 2nd dimension: [piece_id, rotation]

        # Place the pieces on the board with their rotations
        for (t, r, c, a) in solution:
            board[r, c] = [t, a]  # Store piece ID and rotation

        # Check internal edges for matches
        for r in range(n):
            for c in range(m):
                piece_id = board[r, c, 0]
                rotation = board[r, c, 1]

                sides = self.get_sides(piece_id, rotation, pieces_data) # Get the sides of the piece in the current rotation

                # Check right internal edge (c, c+1) only if it's within bounds
                if c < m - 1:  # Internal edge to the right
                    right_piece_id = board[r, c + 1, 0]
                    right_rotation = board[r, c + 1, 1]
                    right_sides = self.get_sides(right_piece_id, right_rotation, pieces_data)
                    if sides[2] == right_sides[0]:  # Right of current piece matches left of right piece
                        shared_edges += 1
                    else:
                        bad_edges += 1

                # Check bottom internal edge (r+1, c) only if it's within bounds
                if r < n - 1:  # Internal edge to the bottom
                    bottom_piece_id = board[r + 1, c, 0]
                    bottom_rotation = board[r + 1, c, 1]
                    bottom_sides = self.get_sides(bottom_piece_id, bottom_rotation, pieces_data)
                    if sides[3] == bottom_sides[1]:  # Bottom of current piece matches top of bottom piece
                        shared_edges += 1
                    else:
                        bad_edges += 1

        # Return shared and bad edge matches
        return shared_edges, bad_edges
    
    # Function to save solution and info to result txt file
    def save_result_to_file(self, solution, n, m, shared_edges, bad_edges, solving_time, filename):
        with open(filename, "w") as file:
            # Write total number of placed pieces
            file.write(f"Total number of placed pieces : {n * m}\n\n")
            file.write("  Table with all pieces :\n\n")
            
            # Write header for columns
            header = "       " + "  ".join([f"{i+1:3}" for i in range(m)]) + "\n\n"
            file.write(header)
            
            # Write each row with placed pieces
            for r in range(n):
                row_pieces = []
                for c in range(m):
                    piece = None
                    for (t, row, col, a) in solution:
                        if row == r and col == c:
                            piece = t
                            break
                    row_pieces.append(f"{piece if piece is not None else 0:3}")
                row_str = f"  {r+1:2}   " + "  ".join(row_pieces) + "\n"
                file.write(row_str)
            
            # Write shared and bad edges
            file.write(f"\n   Shared edges :      {shared_edges}\n")
            file.write(f"   Bad edges    :      {bad_edges}\n")

            # Add the solving time
            file.write(f"\n   Time to solve the puzzle: {solving_time:.2f} seconds\n")

    # Function to log solution details to log txt file
    def log_solution(self, n, m, hint_pieces_count, time_limit, percent_matched, solving_time):
        log_entry = f"{n}x{m}\t{hint_pieces_count}\t{time_limit if time_limit else '-'}\t{percent_matched}%\t{solving_time:.2f}\n"
        with open("log.txt", "a") as log_file:
            log_file.write(log_entry)

# Run the main program
if __name__ == "__main__":
    root = tk.Tk()
    app = EternityIISolverGUI(root)
    root.mainloop()