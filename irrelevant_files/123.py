import pygame, random, os
pygame.init()
pygame.font.init()

def loadImages():
    global images, backImage
    suits = ("C", "D", "H", "S")
    images = [[[0 for x in range(13)] for y in range(4)] for z in range(2)]
    backImage = pygame.image.load(r"Cards\CardBack.png")
    for s in range(4):
        for i in range(13):
            images[0][s][i] = pygame.image.load(r"Cards\\" + str(i + 2) + suits[s] + ".png")
            images[1][s][i] = pygame.image.load(r"Cards\D" + str(i + 2) + suits[s] + ".png")

def newGame():
    # creating game arrays
    global player, curCard, turn, level, deck, gameOver
    deck = list(range(52))
    player = [[[-1 for x in range(5)] for y in range(5)] for z in range(2)]

    for i in range(10):
        r = random.sample(deck, 1)[0]
        deck.remove(r)
        player[(i % 2)][0][i % 5] = r

    gameOver = False
    turn = 1 - (sum(score) % 2)  # shows whose turn it is
    curCard = -1  # current cards are drawn during the loop
    level = 0
    redrawBoard()

def getCPUchoice():
    options = [x for x in range(5) if player[turn][level][x] == -1]
    decision = random.sample(options, 1)[0]
    return decision

def redrawBoard():
    win.fill(backcolor)

    for i in range(5):
        for j in range(level+1):
            if player[0][j][i] > -1:
                if (cpuPlayer is True) and (j == 4):
                    win.blit(backImage,(260 + i * 100, 170 - 30 * j))
                else:
                    win.blit(images[1 - (turn == 0)][player[0][j][i] % 4][player[0][j][i] % 13], (260 + i * 100, 170 - 30 * j))
            if player[1][j][i] > -1:
                win.blit(images[1 - (turn == 1)][player[1][j][i] % 4][player[1][j][i] % 13], (260 + i * 100, 320 + 30 * j))

    textFont = pygame.font.SysFont('comicsansms', 22)
    pygame.draw.rect(win, (150, 200, 150), (50, 250, 160, 100))
    pygame.draw.rect(win, (0, 0, 0), (50, 250, 160, 100),3)
    pygame.draw.line(win,(0, 0, 0), (50, 300),(210,300),3)
    p1 = textFont.render("Player 1 ", False, (0, 0, 250))
    p2 = textFont.render("Player 2 ", False, (250, 0, 0))
    win.blit(p1, (60, 258))
    win.blit(p2, (60, 308))
    s1 = textFont.render(str(score[0]), False, (0, 0, 250))
    s2 = textFont.render(str(score[1]), False, (250, 0, 0))
    if(score[0]//10 > 0):
        win.blit(s1, (180, 258))
    else:
        win.blit(s1, (190, 258))
    if (score[1] // 10 > 0):
        win.blit(s2, (180, 308))
    else:
        win.blit(s2, (190, 308))
    # pygame.draw.rect(win, (150, 200, 150), (870, 510, 100, 60))
    # pygame.draw.rect(win, (0, 0, 0), (870, 510, 100, 60), 3)
    # button = textFont.render("Restart", False, (0, 0, 0))
    # win.blit(button, (882, 522))

def drawEndBoard(winner,labels):
    win.fill(backcolor)
    for i in range(5):
        for j in range(5):
                win.blit(images[1-(winner[i] == 0)][player[0][j][i] % 4][player[0][j][i] % 13],(260 + i * 100, 170 - 30 * j))
                win.blit(images[1-(winner[i] == 1)][player[1][j][i] % 4][player[1][j][i] % 13],(260 + i * 100, 320 + 30 * j))
    textFont = pygame.font.SysFont('comicsansms', 22)
    pygame.draw.rect(win, (150, 200, 150), (50, 250, 160, 100))
    pygame.draw.rect(win, (0, 0, 0), (50, 250, 160, 100), 3)
    pygame.draw.line(win, (0, 0, 0), (50, 300), (210, 300), 3)
    p1 = textFont.render("Player 1 ", False, (0, 0, 250))
    p2 = textFont.render("Player 2 ", False, (250, 0, 0))
    win.blit(p1, (60, 258))
    win.blit(p2, (60, 308))
    s1 = textFont.render(str(score[0]), False, (0, 0, 250))
    s2 = textFont.render(str(score[1]), False, (250, 0, 0))
    if (score[0] // 10 > 0):
        win.blit(s1, (180, 258))
    else:
        win.blit(s1, (190, 258))
    if (score[1] // 10 > 0):
        win.blit(s2, (180, 308))
    else:
        win.blit(s2, (190, 308))
    pygame.draw.rect(win, (150, 200, 150), (870, 510, 100, 60))
    pygame.draw.rect(win, (0, 0, 0), (870, 510, 100, 60), 3)
    button = textFont.render("Restart", False, (0, 0, 0))
    win.blit(button, (882, 522))

    labelFont = pygame.font.SysFont('arial', 12)
    for i in [4,3,2,1,0]: #230 and up or 355 and down
        l = labelFont.render(labels[i], False, (255, 255, 0))
        if winner[i] ==0:
            win.blit(l,(15, 230 - 15*(i-sum(winner[0:i]))))
        else:
            win.blit(l, (15, 355 + 15 * sum(winner[0:i])))

def moveCardTo(turn,row,column):
    n = round(speedModifier*12*(10-column) - 1)
    for i in range(1,n+1):
        redrawBoard()
        if (turn == 0) and (cpuPlayer is True) and (level == 4):
            win.blit(backImage, (780 * ((n - i) / n) + (260 + column * 100) * ((i) / n), 245 * ((n - i) / n) + (170 + 150 * turn + 30 * row * (2 * turn - 1)) * ((i) / n)))
        else:
            win.blit(images[0][curCard % 4][curCard % 13],(780*((n-i)/n) + (260+column*100)*((i)/n), 245*((n-i)/n) + (170+150*turn + 30*row*(2*turn-1))*((i)/n)))
        pygame.display.update()

def calculateWinner():
    hands = ["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"]
    cardNames = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
    handValueCounts = [[[0 for x in range(13)] for y in range(5)] for z in range(2)]
    handSuited =[[0 for x in range(5)] for y in range(2)]
    handValues = [[[0 for x in range(5)] for y in range(5)] for z in range(2)]
    handLabels = [[0 for x in range(5)] for y in range(2)]
    handCode = [[0 for x in range(5)] for y in range(2)]  # first number - hand type, next 2 - hand parameters, last 5 - sorted hand values
    for p in range(2):
        for c in range(5):
            handSuited[p][c] = (player[p][0][c]%4 == player[p][1][c]%4 == player[p][2][c]%4 == player[p][3][c]%4 == player[p][4][c]%4)
            for r in range(5):
                handValueCounts[p][c][player[p][r][c] % 13] += 1
                handValues[p][c][r] = player[p][r][c] % 13

    for p in range(2):
        for c in range(5):
            handValues[p][c].sort(reverse=True)
            m = max(handValueCounts[p][c])
            if (m == 4):
                handLabels[p][c] = hands[7] +" "+ cardNames[handValueCounts[p][c].index(m)] + "s"
                handCode[p][c] = [7,handValueCounts[p][c].index(m),handValueCounts[p][c].index(m)]+handValues[p][c]
            if (m == 3):
                ap = -1 # looking for additional pair, which would give a full house
                for i in range(13):
                    if handValueCounts[p][c][i]==2:
                        ap = i
                if(ap>-1):
                    handLabels[p][c] = hands[6] + " " + cardNames[handValueCounts[p][c].index(m)] + "s and " + cardNames[ap] + "s"
                    handCode[p][c] = [6, handValueCounts[p][c].index(m), ap] + handValues[p][c]
                else:
                    handLabels[p][c] = hands[3] + " " + cardNames[handValueCounts[p][c].index(m)] + "s"
                    handCode[p][c] = [3, handValueCounts[p][c].index(m), handValueCounts[p][c].index(m)] + handValues[p][c]
            if (m == 2):
                tp = [] # looking for two pair
                for i in range(13):
                    if handValueCounts[p][c][i] == 2:
                        tp.append(i)
                if(len(tp)==2):
                    handLabels[p][c] = hands[2] + " " + cardNames[max(tp)] + "s and " + cardNames[min(tp)] + "s"
                    handCode[p][c] = [2, max(tp), min(tp)] + handValues[p][c]
                else:
                    handLabels[p][c] = hands[1] + " of " + cardNames[tp[0]] + "s"
                    handCode[p][c] = [1, tp[0], tp[0]] + handValues[p][c]
            if (m == 1):
                count = 0
                last = False
                l = list(range(13))
                l.insert(0,12)
                for i in l:
                    if handValueCounts[p][c][i] ==1:
                        if last is False:
                            last = True
                            count = 1
                        else:
                            count += 1
                    else:
                        last = False
                if(count==5):
                    if(handSuited[p][c]):
                        handLabels[p][c] = hands[8] + " " + cardNames[max(handValues[p][c])] + " High"
                        handCode[p][c] = [8] + [handValues[p][c][0]] + [handValues[p][c][1]] + handValues[p][c]
                    else:
                        handLabels[p][c] = hands[4] + " " + cardNames[max(handValues[p][c])] + " High"
                        handCode[p][c] = [4] + [handValues[p][c][0]] + [handValues[p][c][1]] + handValues[p][c]
                else:
                    if (handSuited[p][c]):
                        handLabels[p][c] = hands[5] + " " + cardNames[max(handValues[p][c])] + " High"
                        handCode[p][c] = [5] + [handValues[p][c][0]] + [handValues[p][c][1]] + handValues[p][c]
                    else:
                        handLabels[p][c] = hands[0] + " " + cardNames[max(handValues[p][c])] + " High"
                        handCode[p][c] = [0] + [handValues[p][c][0]] + [handValues[p][c][1]] + handValues[p][c]

    winner = [-1 for i in range(5)]
    for c in range(5):
        found = False
        for i in range(8):
            if found is False:
                if handCode[0][c][i] > handCode[1][c][i]:
                    winner[c] = 0
                    found = True
                if handCode[0][c][i] < handCode[1][c][i]:
                    winner[c] = 1
                    found = True

    finalLabels = [-1 for i in range(5)]
    for c in range(5):
        finalLabels[c] = "Won column "+ str(c+1)+ " with "+ str(handLabels[winner[c]][c])

    if(sum(winner) < 3):
        score[0] += 1
    else:
        score[1] += 1

    drawEndBoard(winner, finalLabels)



    #return (handCode + handLabels)

# creating window, setting background color to casino green and naming the window
os.environ['SDL_VIDEO_WINDOW_POS'] = str(100) + "," + str(50)
win = pygame.display.set_mode((1000, 600))
backcolor = (0, 120, 20)
win.fill(backcolor)
pygame.display.set_caption("Five-O Poker")

run = True
score = [0, 0]
speedModifier = 1/3  # smaller means faster
cpuPlayer = True
loadImages()
newGame()

while run:
    pygame.time.delay(10)
    pygame.display.update()

    # should a new card be drawn?
    if (curCard == -1) and (len(deck) > 2):
        curCard = random.sample(deck, 1)[0]
        deck.remove(curCard)
        redrawBoard()
        if (turn == 0) and (cpuPlayer is True) and (level == 4):
            win.blit(backImage, (780, 245))
        else:
            win.blit(images[0][curCard % 4][curCard % 13], (780, 245))

    # has the board been filled?
    if len(deck) == 2:
        curCard = random.sample(deck, 1)[0]
        deck.remove(curCard)
        gameOver = True
        calculateWinner()

    # is this the last card of the row? (if so, it'll be auto-placed)
    if (len(deck) - 2) % 10 <= 1:
        pygame.display.update()
        pygame.time.delay(round(200 * speedModifier))
        choice = -1
        for i in range(5):
            if player[turn][level][i] == -1:
                choice = i
        moveCardTo(turn, level, choice)
        player[turn][level][choice] = curCard
        curCard = -1
        turn = 1 - turn

    # is the cpu on?
    if (cpuPlayer is True) and ((len(deck)-2) % 10 > 1) and (turn == 0) and (len(deck) > 2):
        if(len(deck)%10 == 1):
            level += 1
        pygame.display.update()
        pygame.time.delay(round(200 * speedModifier))
        choice = getCPUchoice()
        moveCardTo(turn, level, choice)
        player[turn][level][choice] = curCard
        curCard = -1
        turn = 1 - turn

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if not gameOver:
            # placing cards based on mouse click
            if event.type == pygame.MOUSEBUTTONUP and ((len(deck)-2) % 10 > 1) and (cpuPlayer*(1-turn) != 1):
                pos = pygame.mouse.get_pos()
                if ((pos[0]) in range(260,740)) and ((pos[0]-260) % 100 in range(0,80)) and (pos[1]+270*(turn == 0) in range(320,550)):
                    choice = (pos[0] - 260)//100
                    level = 4 - ((len(deck)-2) // 10)
                    if player [turn][level][choice] == -1:
                        moveCardTo(turn, level, choice)
                        player[turn][level][choice] = curCard  # make sure to do the addition of the card after the animation
                        curCard=-1
                        turn = 1-turn
            # placing cards based on use of 1-5 keys
            if event.type == pygame.KEYDOWN and ((len(deck) - 2) % 10 > 1) and (cpuPlayer*(1-turn) != 1):
                if event.key == pygame.K_1:
                    choice = 0
                    level = 4 - ((len(deck) - 2) // 10)
                    if player[turn][level][choice] == -1:
                        moveCardTo(turn, level, choice)
                        player[turn][level][choice] = curCard  # make sure to do the addition of the card after the animation
                        curCard = -1
                        turn = 1 - turn
                if event.key == pygame.K_2:
                    choice = 1
                    level = 4 - ((len(deck) - 2) // 10)
                    if player[turn][level][choice] == -1:
                        moveCardTo(turn, level, choice)
                        player[turn][level][choice] = curCard  # make sure to do the addition of the card after the animation
                        curCard = -1
                        turn = 1 - turn
                if event.key == pygame.K_3:
                    choice = 2
                    level = 4 - ((len(deck) - 2) // 10)
                    if player[turn][level][choice] == -1:
                        moveCardTo(turn, level, choice)
                        player[turn][level][choice] = curCard  # make sure to do the addition of the card after the animation
                        curCard = -1
                        turn = 1 - turn
                if event.key == pygame.K_4:
                    choice = 3
                    level = 4 - ((len(deck) - 2) // 10)
                    if player[turn][level][choice] == -1:
                        moveCardTo(turn, level, choice)
                        player[turn][level][choice] = curCard  # make sure to do the addition of the card after the animation
                        curCard = -1
                        turn = 1 - turn
                if event.key == pygame.K_5:
                    choice = 4
                    level = 4 - ((len(deck) - 2) // 10)
                    if player[turn][level][choice] == -1:
                        moveCardTo(turn, level, choice)
                        player[turn][level][choice] = curCard  # make sure to do the addition of the card after the animation
                        curCard = -1
                        turn = 1 - turn
        else:
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                if (pos[0] in range(870, 970)) and (pos[1] in range(510, 570)):
                    newGame()

pygame.quit()
