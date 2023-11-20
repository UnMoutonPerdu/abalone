from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from enum import Enum
import numpy as np
import random

WHITE = 1
BLACK = -1
QUIESCENT_DEPTH = 2
MAX_DEPTH = 3

# Zobrist Hashing pour les Tables de Transposition utilisee dans nos algos
class ZobristHashing:
    def __init__(self, board_size, num_pieces):
        self.board_size = board_size
        self.num_players = num_pieces
        self.hash_table = [[random.getrandbits(64) for _ in range(num_pieces)] for _ in range(board_size)]
        self.transposition_table = {}
        #self.black_turn = random.getrandbits(64)

    def calculate_hash(self, current_state: GameState, color):
        grid = current_state.get_rep().get_grid()
        hash_value = 0
        for row in range(len(grid)):
            for col in range(len(grid[row])):
                piece = grid[row][col]
                index_square = 9*row + col 
                if piece == 'W':
                    hash_value ^= self.hash_table[index_square][0]
                elif piece == 'B':
                    hash_value ^= self.hash_table[index_square][1]
        return hash_value

# Initialisation de la TT pour l'algo
zobrist = ZobristHashing(81, 2)

class MyPlayer(PlayerAbalone):
    """
    Player class for Abalone game.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "bob", time_limit: float=60*15,*args) -> None:
        """
        Initialize the PlayerAbalone instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type,name,time_limit,*args)

    def euclidean_distance(self, P1, P2):
        return np.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2)

    # Cette fonction provient du fichier master_abalone.py que nous avons modifiée pour pouvoir l'utiliser correctement qu'import les cases sélectionnées
    def manhattan_dist(self, A, B):
            mask1 = [(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8)]
            mask2 = [(0,4)]
            diff = (abs(B[0] - A[0]),abs(B[1] - A[1]))
            dist = (abs(B[0] - A[0]) + abs(B[1] - A[1]))/2
            if diff in mask1:
                dist += 1
            if diff in mask2:
                dist += 2
            return dist

    # La fonction score qui retourne l'écart entre le nombre de billes renversées par les blancs et le nombre de billes renversées par les noirs.
    def score_function(self, current_state: GameState):
        scores = list(current_state.get_scores().values())
        return scores[0] - scores[1]
    
    # Fonction qui filtre les actions possibles pour ne garder que celle ou il n'y a pas de suicide.
    def no_suicidal_action(self, color, current_state: GameState, possible_actions: list[Action]) -> list[Action]:
        filtered_possible_actions = []
        for action in possible_actions:
            new_state = action.get_next_game_state()
            current_scores = list(current_state.get_scores().values())
            next_scores = list(new_state.get_scores().values())
            if color == WHITE: 
                if current_scores[0] == next_scores[0]: 
                    filtered_possible_actions.append(action)
            elif color == BLACK:
                if current_scores[1] == next_scores[1]:
                    filtered_possible_actions.append(action)
        
        return filtered_possible_actions

    # Fonction qui retourne True si l'etat peut être considere comme stable
    def is_quiescent(self, current_state: GameState, next_state: GameState):
        current_scores = list(current_state.get_scores().values())
        next_scores = list(next_state.get_scores().values())
        if (current_scores[0] == next_scores[0] and current_scores[1] == next_scores[1]):
            return True
        return False

    # Notre fonction heuristique
    def heuristic(self, current_state: GameState):
        # On récupère la grille du plateau
        grid = current_state.get_rep().get_env()
        iB, iW, jB, jW, count_B, count_W = 0, 0, 0, 0, 0, 0

        # On compte le nombre de billes noires et blanches sur le plateau et on récupère leurs positions
        for elem in grid:
            if grid[elem].get_type() == 'W':
                iW += elem[0]
                jW += elem[1]
                count_W += 1
            elif grid[elem].get_type() == 'B':
                iB += elem[0]
                jB += elem[1]
                count_B += 1
        
        # On calcule les centres de masses des 2 groupes
        center_mass_W = np.array((iW, jW))/count_W
        center_mass_B = np.array((iB, jB))/count_B

        # On considère egalement le centre auquel on attribue un certain poids
        center        = np.array((8, 4))
        center_weight = 28

        # On calcule les coordonées d'un point R qui est un point de coordonnées pondérées par les coordonées du centre du plateau et des différentes centres de masse. Ce point R est différent du centre et force le joueur à gagner le centre tout en repoussant l'équipe adverse.
        total_count = count_W + count_B + center_weight
        coord_R = center_mass_W*(count_W/total_count) + center_mass_B*(count_B/total_count) + center*(center_weight/total_count)

        # On calcule la variance de chaque groupe pour quantifier leur degré de cohésion, ainsi que leur distance au centre R.
        sum_dist_W, sum_dist_B = 0, 0
        variance_W, variance_B = 0, 0
        for elem in grid:
            if grid[elem].get_type() == 'W':
                sum_dist_W += self.euclidean_distance(elem, coord_R)
                variance_W += self.euclidean_distance(elem, center_mass_W)**2
            elif grid[elem].get_type() == 'B':
                sum_dist_B += self.euclidean_distance(elem, coord_R)
                variance_B += self.euclidean_distance(elem, center_mass_B)**2

        center_distance = sum_dist_B/count_B - sum_dist_W/count_W
        cohesion = np.sqrt(variance_B)/count_B - np.sqrt(variance_W)/count_W
        marbles = self.score_function(current_state)

        return  center_distance + cohesion + 100*marbles

    def alpha_beta_search(self, alpha, beta, color, depth, max_depth, current_state: GameState, heuristic=heuristic):

        ### TT + Fin de jeu ###

        # On calcule la hash_value de notre etat
        new_entry = {'best_action': None, 'best_score': None, 'flag': None, 'depth': None}
        hash_value = zobrist.calculate_hash(current_state, color)

        # On regarde si on a deja vu l'etat et si la valeur anciennement trouvée est interessante
        if hash_value in zobrist.transposition_table and zobrist.transposition_table[hash_value]['depth'] <= depth:
            new_entry = zobrist.transposition_table[hash_value]
            best_action, best_score, flag = new_entry['best_action'], new_entry['best_score'], new_entry['flag']

            if flag == 'exact':
                return best_score, action
            elif flag == 'lower':
                alpha = max(alpha, best_score)
            elif flag == 'upper':
                beta = min(beta, best_score)

            if alpha >= beta:
                return best_score, best_action

        # Si on est a la fin de l'alpha-beta ou du jeu on retourne les valeurs de score et d'etat
        if depth == max_depth or current_state.is_done():
            return heuristic(self, current_state), current_state

        ### Alpha - Beta ###

        # Setup des valeurs
        best_action = None
        possible_actions = list(current_state.get_possible_actions())
        possible_actions = self.no_suicidal_action(color, current_state, possible_actions)
        #random.shuffle(possible_actions) # Random shuffle pour ameliorer les performances de l'algo
        possible_actions = sorted(possible_actions, key=lambda x: -color*heuristic(self, x.get_next_game_state()))
        #possible_actions = set(possible_actions)
        best_score = (-color)*np.Inf

        # On parcourt chaque action possible
        for action in possible_actions:
            new_state = action.get_next_game_state()
            new_max_depth = max_depth
            if max_depth == MAX_DEPTH and (max_depth - depth == 1) and (not self.is_quiescent(current_state, new_state)):
                new_max_depth += QUIESCENT_DEPTH

            new_score, _ = self.alpha_beta_search(alpha, beta, -color, depth+1, new_max_depth, new_state, heuristic)

            if color == WHITE:
                if new_score > best_score:
                    best_score = new_score
                    alpha = max(alpha, best_score)
                    best_action = action
                if best_score >= beta:
                    break

            if color == BLACK:
                if new_score < best_score:
                    best_score = new_score
                    beta = min(beta, best_score)
                    best_action = action
                if best_score <= alpha:
                    break

        # On rajoute une entrée à notre TT
        new_entry['best_score'] = best_score
        new_entry['best_action'] = best_action
        new_entry['depth'] = depth
        if best_score <= alpha:
            new_entry['flag'] = 'upper'
        elif best_score >= beta:
            new_entry['flag'] = 'lower'  
        else:
            new_entry['flag'] = 'exact' 
        
        zobrist.transposition_table[hash_value] = new_entry
        return best_score, best_action

    # Fonction de choix de l'action lors du jeu
    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """

        if self.piece_type == "W":
            color = WHITE
        if self.piece_type == "B":
            color = BLACK

        zobrist.transposition_table = {}

        score, action = self.alpha_beta_search(-np.Inf, np.Inf, color, 0, MAX_DEPTH, current_state)
        scores = list(current_state.get_scores().values())

        print("Score: ", score)
        print("Step: ", current_state.get_step())
        print("Fallen White Marbles: ", -scores[0])
        print("Fallen Black Marbles: ", -scores[1])
        if action == None:
            print("No Possible Action")
        return action
