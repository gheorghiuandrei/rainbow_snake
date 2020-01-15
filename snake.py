import collections
import itertools
import random
import pygame

GRID_SIZE = 16
BOX_SIZE = 16
WAIT = 200


class Box(pygame.sprite.Sprite):
    def __init__(self, position):
        super().__init__()

        self.color = pygame.Color("white")
        self.rects = {
            (0, -1): (BOX_SIZE / 4, 0, BOX_SIZE / 2, BOX_SIZE / 4),
            (0, 1): (BOX_SIZE / 4, BOX_SIZE * 3 / 4, BOX_SIZE / 2, BOX_SIZE),
            (-1, 0): (0, BOX_SIZE / 4, BOX_SIZE / 4, BOX_SIZE / 2),
            (1, 0): (BOX_SIZE * 3 / 4, BOX_SIZE / 4, BOX_SIZE, BOX_SIZE / 2),
        }
        self.image = pygame.Surface((BOX_SIZE,) * 2, pygame.SRCALPHA)
        self.image.fill(self.color, ((BOX_SIZE / 4,) * 2, (BOX_SIZE / 2,) * 2))
        self.rect = self.image.get_rect()
        self.rect.topleft = pygame.math.Vector2(position) * BOX_SIZE
        self.position = tuple(position)

    def fill(self, key):
        self.image.fill(self.color, self.rects[tuple(key)])


class Snake:
    def __init__(self):
        self.directions = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        self.direction = pygame.math.Vector2(1, 0)
        self.body = collections.deque()
        self.sprites = pygame.sprite.Group()
        self.shed_box = None

        for i in range(4):
            box = Box((GRID_SIZE / 4 - i, GRID_SIZE / 2))
            box.fill(-self.direction)

            if i == 0:
                self.head = box
            else:
                box.fill(self.direction)
                self.body.append(box)

            self.sprites.add(box)

    def move(self, action):
        if all(
            i != j for i, j in zip(self.direction, self.directions[action])
        ):
            self.direction.xy = self.directions[action]

    def update(self):
        self.shed_box = self.body.pop()
        self.sprites.remove(self.shed_box)
        self.head.fill(self.direction)
        self.body.appendleft(self.head)
        self.head = Box(self.head.position + self.direction)
        self.head.fill(-self.direction)
        self.sprites.add(self.head)

    def eat(self):
        self.body.append(self.shed_box)
        self.sprites.add(self.shed_box)


class Apple:
    def __init__(self):
        self.sprite = pygame.sprite.GroupSingle()
        self.all_boxes = set(itertools.product(range(GRID_SIZE), repeat=2))

    def spawn(self, snake):
        full_boxes = {snake.head.position} | {
            box.position for box in snake.body
        }
        empty_boxes = tuple(self.all_boxes - full_boxes)

        if len(empty_boxes) != 0:
            self.sprite.add(Box(random.choice(empty_boxes)))
            done = False
        else:
            done = True

        return done


class Environment:
    def __init__(self):
        self.screen = pygame.display.set_mode((GRID_SIZE * BOX_SIZE,) * 2)
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.actions = {
            pygame.K_UP: 0,
            pygame.K_DOWN: 1,
            pygame.K_LEFT: 2,
            pygame.K_RIGHT: 3,
        }
        self.done = None
        self.snake = None
        self.apple = None

    def reset(self):
        self.done = False
        self.snake = Snake()
        self.apple = Apple()
        self.apple.spawn(self.snake)
        frame = self._update()

        return frame

    def step(self, action=None):
        if action is not None:
            self.snake.move(action)
        else:
            for event in pygame.event.get():
                if (
                    event.type == pygame.QUIT
                    or event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE
                ):
                    self.done = True
                elif (
                    event.type == pygame.KEYDOWN and event.key in self.actions
                ):
                    self.snake.move(self.actions[event.key])
                    break

        reward = 0
        self.snake.update()

        if pygame.sprite.spritecollideany(self.snake.head, self.apple.sprite):
            reward += 1
            self.snake.eat()
            self.done = self.apple.spawn(self.snake)
        elif (
            pygame.sprite.spritecollideany(self.snake.head, self.snake.body)
            or any(i < 0 for i in self.snake.head.position)
            or any(i == GRID_SIZE for i in self.snake.head.position)
        ):
            reward -= 1
            self.done = True

        frame = self._update()

        return reward, frame, self.done

    def _update(self):
        self.screen.fill(pygame.Color("black"))
        self.snake.sprites.draw(self.screen)
        self.apple.sprite.draw(self.screen)
        pygame.display.update()
        self.clock.tick()
        frame = pygame.surfarray.array3d(pygame.display.get_surface())

        return frame


def main():
    environment = Environment()
    environment.reset()

    while not environment.done:
        environment.step()
        pygame.time.delay(WAIT)


if __name__ == "__main__":
    main()
