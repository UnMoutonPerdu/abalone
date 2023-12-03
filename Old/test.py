from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from enum import Enum
import numpy as np
import random

WHITE = 1
BLACK = -1

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

    def score_function(self, current_state: GameState):
        scores = list(current_state.get_scores().values())
        return scores[0] - scores[1]
    
    def geometric_heuristic(self, current_state: GameState):
        grid = current_state.get_rep().get_grid()
        iB, iW, jB, jW, count_B, count_W = 0, 0, 0, 0, 0, 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 'W':
                    iW += i
                    jW += j
                    count_W += 1
                if grid[i][j] == 'B':
                    iB += i
                    jB += j
                    count_B += 1
        
        center_mass_W = np.array((iW, jW))/count_W
        center_mass_B = np.array((iB, jB))/count_B
        center        = np.array((4, 4))
        center_weight = 3

        total_count = count_W + count_B + center_weight
        coord_R = center_mass_W*(count_W/total_count) + center_mass_B*(count_B/total_count) + center*(center_weight/total_count)

        sum_dist_W, sum_dist_B = 0, 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 'W':
                    sum_dist_W += self.euclidean_distance((i,j),coord_R)
                if grid[i][j] == 'B':
                    sum_dist_B += self.euclidean_distance((i,j),coord_R)

        return 1/(sum_dist_W+1) - 1/(sum_dist_B+1) + 5*self.score_function(current_state)
        #return sum_dist_B - sum_dist_W + 5*self.score_function(current_state)

    def alpha_beta_search(self, alpha, beta, color, depth, max_depth, current_state: GameState, heuristic=geometric_heuristic):

        if depth == max_depth or current_state.is_done():
            return heuristic(self, current_state), current_state

        best_action = None
        possible_actions = list(current_state.get_possible_actions())
        random.shuffle(possible_actions)
        possible_actions = set(possible_actions)
        best_score = (-color)*np.Inf

        for action in possible_actions:
            new_state = action.get_next_game_state()
            new_score, _ = self.alpha_beta_search(alpha, beta, -color, depth+1, max_depth, new_state, heuristic)

            if color == WHITE:
                if new_score > best_score:
                    best_score = new_score
                    alpha = max(alpha, best_score)
                    best_action = action
                if best_score >= beta:
                    return best_score, best_action

            if color == BLACK:
                if new_score < best_score:
                    best_score = new_score
                    beta = min(beta, best_score)
                    best_action = action
                if best_score <= alpha:
                    return best_score, best_action

        return best_score, best_action

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """
        """ grid = current_state.get_rep().get_grid()
        print(grid)

        print(current_state.get_rep().get_neighbours(0,4))

        print(current_state.get_rep().get_env())
        print(current_state.get_rep().get_env()[(0,4)].get_type())
        print(len(current_state.get_rep().get_env()))

        print(list(current_state.get_rep().get_env().keys()))

        for elem in current_state.get_rep().get_env():
            print(elem)

        def manhattanDist(A, B):
            mask1 = [(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8)]
            mask2 = [(0,4)]
            diff = (abs(B[0] - A[0]),abs(B[1] - A[1]))
            dist = (abs(B[0] - A[0]) + abs(B[1] - A[1]))/2
            if diff in mask1:
                dist += 1
            if diff in mask2:
                dist += 2
            return dist
        
        print(manhattanDist((5,1), (8,6))) """

        current_scores = list(current_state.get_scores().values())

        print(current_scores)

        grid = current_state.get_rep().get_env()

        print(current_state.get_rep().get_env())

        print(current_state.get_neighbours(grid[0]))

        dico = current_state.get_neighbours(4, 0)

        print(sum([dico[side][0] == 'B' for side in dico]))

        print(grid[(4, 0)].get_type())

        for elem in dico:
            print(dico[elem][0])

        """ if self.piece_type == "W":
            color = WHITE
        if self.piece_type == "B":
            color = BLACK
        
        score, action = self.alpha_beta_search(-np.Inf, np.Inf, color, 0, 3, current_state)
        scores = list(current_state.get_scores().values())

        print("Score: ", score)
        print("Step: ", current_state.get_step())
        print("Fallen White Marbles: ", -scores[0])
        print("Fallen Black Marbles: ", -scores[1])
        if action == None:
            print("No Possible Action") """
        return None
