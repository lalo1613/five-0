# Simulating straight odds
straightOdds <- function(iterations=1000, additional_cards=NULL, amounts, n)
{
  # the function uses letters as card values, with the column's first card being 'e' and the next 4 cards to the left
  # and to the right being 'a-d' and 'f-i' respectively. 
  
  cards_at_hand <- "e" 
  if(!is.null(additional_cards))
  {
    if(abs(additional_cards)>4)
      return(0)
    cards_at_hand <- c(cards_at_hand,letters[5+additional_cards])
  }
  
  # simple test to eliminate posibility of straight being unachivable  
  lwr <- which(letters==min(cards_at_hand))
  upr <- which(letters==max(cards_at_hand))
  if(any( (amounts[lwr:upr]==0) & !(lwr:upr %in% which(letters %in% cards_at_hand)) ))
    return(0)
  
  for(i in 2:4) # the goal here is to prevent us from "choosing" a card from an unachivable straight (i.e taking "i" when amount of "g" is 0) 
  {
    if((amounts[4+i]==0) && !(4+i %in% additional_cards))
      amounts[(5+i):9] <- 0
    if((amounts[6-i]==0) && !(6-i %in% additional_cards))
      amounts[(5-i):1] <- 0
  }
  
  cards_left <- 5 - length(cards_at_hand)
  
  if((n-2) %/% 10 < cards_left) # see 'first round cards seen' in ...
    first_round_cards_seen <- ceiling((((n-2)%%10 + 1)%/%2)*0.5 + 0.5)
  else
    first_round_cards_seen <- 3
  
  amounts[c(5,5+additional_cards)] <- 0 # making sure we don't try to get any of the cards we already have
  successes <- 0
  
  for (k in 1:iterations)
  {
    cards_at_hand <- "e"
    if(!is.null(additional_cards))
      cards_at_hand <- c(cards_at_hand,letters[5+additional_cards])
    needed <- (amounts>0)
    
    lwr <- which(letters==min(cards_at_hand))
    upr <- which(letters==max(cards_at_hand))
    needed <- ((abs(1:9 - lwr) < 5) & (abs(1:9 - upr) < 5) & needed)
    
    amounts[!needed] <- 0
    #by this point we've fixed 'amounts' and 'needed' vecs to only the relevant cards
    
    vec <- rep("z",n-sum(amounts)) # vec serves as the deck
    r_cards <- list() # list containing the 'cards' drawn each round
    
    for(i in 1:length(amounts))
    {
      vec <- c(vec,rep(letters[i],amounts[i]))
    }
    
    for(i in 1:cards_left)
    {
      if(i==1)
      {
        temp <- sample(1:length(vec),first_round_cards_seen)
      }
      else
      {
        temp <- sample(1:length(vec),3)
      }
      r_cards[[i]] <- vec[temp]
      vec <- vec[-temp]
    }
    
    over <- F
    for(i in 1:cards_left)
    {
      found <- F
      for(j in 1:length(r_cards[[i]]))
      {
        if ((!over)&&(!found)&&(r_cards[[i]][[j]] %in% letters[which(needed)]))
        {
          found <- T
          cards_at_hand[1+length(cards_at_hand)] <- r_cards[[i]][[j]]
          needed[which(letters==r_cards[[i]][[j]])] <- F
        }
      }
      if(!over)
      {
        if(found)
        {
          lwr <- which(letters==min(cards_at_hand))
          upr <- which(letters==max(cards_at_hand))
          needed <- ((abs(1:9 - lwr) < 5) & (abs(1:9 - upr) < 5) & needed)
        }
        else
        {
          over <- T
        }
      }
    }
    successes <- successes + !over
  }
  successes/iterations
}
# Calculating flush odds
flushOdds <- function(amount_at_hand,cards_remaining,n)
{
  rounds_left <- 5 - amount_at_hand
  
  if((n-2) %/% 10 < rounds_left)
    first_round_cards_seen <- ceiling((((n-2)%%10 + 1)%/%2)*0.5 + 0.5)
  else
    first_round_cards_seen <- 3
  
  sum(unlist(flushRecursive(rounds_left,cards_remaining,n,first_round_cards_seen)))
}

flushRecursive <- function(rl,cr,n,crk) #rounds left, cards remaining, cards left in deck, cards seen this round
{
  if(rl==1)
  {
    i <- 1:min(crk,cr)
    return (phyper(i,cr,n-cr,crk) - phyper(i-1,cr,n-cr,crk))
  }
  else
  {
    prList <- list()
    for(i in 1:min(crk,cr-rl+1))
    {
      prList[[i]] <- (phyper(i,cr,n-cr,crk) - phyper(i-1,cr,n-cr,crk))*unlist(flushRecursive(rl-1,cr-i,n-crk,3))
    }
    return(prList)
  }
}

pairingOdds_2unequal <- function(cr_a,cr_b,n,others_remaining)
{
  frcs <- ifelse(((n-2) %/% 10 < 3),ceiling((((n-2)%%10 + 1)%/%2)*0.5 + 0.5),3)
  others_probs <- ((2:4)*others_remaining[2:4])/sum((1:4)*others_remaining)
  
  p_na <- phyper(0,cr_a,n-cr_a,frcs+6) # no pair for a
  p_nb <- phyper(0,cr_b,n-cr_b,frcs+6) # no pair for b
  p_na_and_nb <-phyper(0,cr_a + cr_b,n - cr_a - cr_b,frcs+6) # no pair for neither (could still be additional pair)
  p_a_and_b <- (1 - p_na - p_nb + p_na_and_nb)*0.75 # probability of seeing both, not necesarily in different rounds. 0.75 is aprox. fix
  
  if(cr_a ==3) # at least 3 of a kind a (could be FH, 4oK) and 4 of a kind a
  {
    if(frcs>1)
    {
      p_3a <- (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*(phyper(1,2,n-3-frcs+1,3) - phyper(0,2,n-3-frcs+1,3))+ #11x
        (phyper(2,3,n-3,frcs) - phyper(1,3,n-3,frcs))*(phyper(1,1,n-3-frcs+2,3) - phyper(0,1,n-3-frcs+2,3))+ #210
        (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*(phyper(2,2,n-3-frcs+1,3) - phyper(1,2,n-3-frcs+1,3))+ #120
        phyper(0,3,n-3,frcs)*(phyper(1,3,n-3-frcs,3) - phyper(0,3,n-3-frcs,3))*(1 - phyper(0,2,n-5-frcs,3))+ # 01(1/2)
        phyper(0,3,n-3,frcs)*(phyper(2,3,n-3-frcs,3) - phyper(1,3,n-3-frcs,3))*(phyper(1,1,n-4-frcs,3) - phyper(0,1,n-4-frcs,3))+ # 021
        (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*phyper(0,2,n-3-frcs+1,3)*(1 - phyper(0,2,n-6-frcs+1,3))+ # 10(1/2)
        (phyper(2,3,n-3,frcs) - phyper(1,3,n-3,frcs))*phyper(0,1,n-3-frcs+2,3)*(phyper(1,1,n-6-frcs+2,3) - phyper(0,1,n-6-frcs+2,3)) # 201
    }
    if(frcs==1)
    {
      p_3a <- (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*(phyper(1,2,n-3-frcs+1,3) - phyper(0,2,n-3-frcs+1,3))+ #11x
        (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*(phyper(2,2,n-3-frcs+1,3) - phyper(1,2,n-3-frcs+1,3))+ #120
        phyper(0,3,n-3,frcs)*(phyper(1,3,n-3-frcs,3) - phyper(0,3,n-3-frcs,3))*(1 - phyper(0,2,n-5-frcs,3))+ # 01(1/2)
        phyper(0,3,n-3,frcs)*(phyper(2,3,n-3-frcs,3) - phyper(1,3,n-3-frcs,3))*(phyper(1,1,n-4-frcs,3) - phyper(0,1,n-4-frcs,3))+ # 021
        (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*phyper(0,2,n-3-frcs+1,3)*(1 - phyper(0,2,n-6-frcs+1,3)) # 10(1/2)
    }
    p_4a <- (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*(phyper(1,2,n-3-frcs+1,3) - phyper(0,2,n-3-frcs+1,3))*(phyper(1,1,n-5-frcs+1,3) - phyper(0,1,n-5-frcs+1,3)) # 4 of a kind a
  }
  if(cr_a == 2)# at least 3 of a kind a (could be FH)
  {
    p_3a <- (phyper(1,2,n-2,frcs) - phyper(0,2,n-2,frcs))*(phyper(1,1,n-2-frcs+1,3) - phyper(0,1,n-2-frcs+1,3))+ #11x
      phyper(0,2,n-2,frcs)*(phyper(1,2,n-2-frcs,3) - phyper(0,2,n-2-frcs,3))*(1 - phyper(0,1,n-4-frcs,3))+ # 011
      (phyper(1,2,n-2,frcs) - phyper(0,2,n-2,frcs))*phyper(0,1,n-2-frcs+1,3)*(1 - phyper(0,1,n-5-frcs+1,3)) # 101
    p_4a <- 0
  }
  if(cr_a < 2)
  {
    p_3a <- 0
    p_4a <- 0
  }
  
  if(cr_b ==3) # at least 3 of a kind b (could be FH, 4oK) and 4 of a kind b
  {
    if(frcs>1)
    {
      p_3b <- (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*(phyper(1,2,n-3-frcs+1,3) - phyper(0,2,n-3-frcs+1,3))+ #11x
        (phyper(2,3,n-3,frcs) - phyper(1,3,n-3,frcs))*(phyper(1,1,n-3-frcs+2,3) - phyper(0,1,n-3-frcs+2,3))+ #210
        (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*(phyper(2,2,n-3-frcs+1,3) - phyper(1,2,n-3-frcs+1,3))+ #120
        phyper(0,3,n-3,frcs)*(phyper(1,3,n-3-frcs,3) - phyper(0,3,n-3-frcs,3))*(1 - phyper(0,2,n-5-frcs,3))+ # 01(1/2)
        phyper(0,3,n-3,frcs)*(phyper(2,3,n-3-frcs,3) - phyper(1,3,n-3-frcs,3))*(phyper(1,1,n-4-frcs,3) - phyper(0,1,n-4-frcs,3))+ # 021
        (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*phyper(0,2,n-3-frcs+1,3)*(1 - phyper(0,2,n-6-frcs+1,3))+ # 10(1/2)
        (phyper(2,3,n-3,frcs) - phyper(1,3,n-3,frcs))*phyper(0,1,n-3-frcs+2,3)*(phyper(1,1,n-6-frcs+2,3) - phyper(0,1,n-6-frcs+2,3)) # 201
    }
    if(frcs==1)
    {
      p_3b <- (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*(phyper(1,2,n-3-frcs+1,3) - phyper(0,2,n-3-frcs+1,3))+ #11x
        (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*(phyper(2,2,n-3-frcs+1,3) - phyper(1,2,n-3-frcs+1,3))+ #120
        phyper(0,3,n-3,frcs)*(phyper(1,3,n-3-frcs,3) - phyper(0,3,n-3-frcs,3))*(1 - phyper(0,2,n-5-frcs,3))+ # 01(1/2)
        phyper(0,3,n-3,frcs)*(phyper(2,3,n-3-frcs,3) - phyper(1,3,n-3-frcs,3))*(phyper(1,1,n-4-frcs,3) - phyper(0,1,n-4-frcs,3))+ # 021
        (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*phyper(0,2,n-3-frcs+1,3)*(1 - phyper(0,2,n-6-frcs+1,3)) # 10(1/2)
    }
    p_4b <- (phyper(1,3,n-3,frcs) - phyper(0,3,n-3,frcs))*(phyper(1,2,n-3-frcs+1,3) - phyper(0,2,n-3-frcs+1,3))*(phyper(1,1,n-5-frcs+1,3) - phyper(0,1,n-5-frcs+1,3)) # 4 of a kind a
  }
  if(cr_b == 2)# at least 3 of a kind b (could be FH)
  {
    p_3b <- (phyper(1,2,n-2,frcs) - phyper(0,2,n-2,frcs))*(phyper(1,1,n-2-frcs+1,3) - phyper(0,1,n-2-frcs+1,3))+ #11x
      phyper(0,2,n-2,frcs)*(phyper(1,2,n-2-frcs,3) - phyper(0,2,n-2-frcs,3))*(1 - phyper(0,1,n-4-frcs,3))+ # 011
      (phyper(1,2,n-2,frcs) - phyper(0,2,n-2,frcs))*phyper(0,1,n-2-frcs+1,3)*(1 - phyper(0,1,n-5-frcs+1,3)) # 101
    p_4b <- 0
  }
  if(cr_b < 2)
  {
    p_3b <- 0
    p_4b <- 0
  }
  
  p_fh_3a_2b <- (phyper(1,cr_b,n-cr_b,3)-phyper(0,cr_b,n-cr_b,3))*p_3a #not the actual odds, lazy aproximation
  p_fh_3b_2a <- (phyper(1,cr_a,n-cr_a,3)-phyper(0,cr_a,n-cr_a,3))*p_3b #not the actual odds, lazy aproximation
  p_only_2a_2b <- p_a_and_b - p_fh_3a_2b - p_fh_3b_2a
  
  p_a_and_o <- (p_nb - p_na_and_nb - (p_3a - p_fh_3a_2b))*(sum(others_probs*(1-phyper(0,1:3,n- cr_a - cr_b - 1:3,3)))) # 2 pair A + O
  p_only2_a <- (p_nb - p_na_and_nb - (p_3a - p_fh_3a_2b))*(1-sum(others_probs*(1-phyper(0,1:3,n- cr_a - cr_b - 1:3,3)))) # pair A
  p_b_and_o <- (p_na - p_na_and_nb - (p_3b - p_fh_3b_2a))*(sum(others_probs*(1-phyper(0,1:3,n- cr_a - cr_b - 1:3,3)))) # 2 pair B + O
  p_only2_b <- (p_na - p_na_and_nb - (p_3b - p_fh_3b_2a))*(1-sum(others_probs*(1-phyper(0,1:3,n- cr_a - cr_b - 1:3,3)))) # pair B
  p_3o <- (p_na_and_nb)*
    (others_probs[2]*(phyper(1,2,n- cr_a - cr_b - 2,3)-phyper(0,2,n- cr_a - cr_b - 2,3))*(phyper(1,1,n-2- cr_a - cr_b - 4,3)-phyper(0,1,n-2- cr_a - cr_b - 4,3))+
       others_probs[3]*(phyper(2,3,n- cr_a - cr_b - 2,3)-phyper(1,3,n- cr_a - cr_b - 2,3))*(phyper(1,1,n-1- cr_a - cr_b - 3,3)-phyper(0,1,n-1- cr_a - cr_b - 3,3))+
       others_probs[3]*(phyper(1,3,n- cr_a - cr_b - 2,3)-phyper(0,3,n- cr_a - cr_b - 2,3))*(1-phyper(0,2,n-2- cr_a - cr_b - 4,3)))
  p_only2_o <- (p_na_and_nb)*(sum(others_probs*(1-phyper(0,1:3,n- cr_a - cr_b - 1:3,6))) + sum(others_probs*(1-phyper(0,1:3,n- cr_a - cr_b - 1:3,3))) - 2*p_3o)
  p_only3_a <- p_3a - p_4a - p_fh_3a_2b
  p_only3_b <- p_3b - p_4b - p_fh_3b_2a
  
  {
    vec_fix <- (1 - p_na - p_nb + p_na_and_nb)*0.25*c(p_only2_a,p_only2_b,p_a_and_o,p_b_and_o,p_only3_a,p_only3_b,p_4a,p_4b)/sum(c(p_only2_a,p_only2_b,p_a_and_o,p_b_and_o,p_only3_a,p_only3_b,p_4a,p_4b))
    p_only2_a <- p_only2_a+vec_fix[1]
    p_only2_b <- p_only2_b+vec_fix[2]
    p_a_and_o <- p_a_and_o+vec_fix[3]
    p_b_and_o <- p_b_and_o+vec_fix[4]
    p_only3_a <- p_only3_a+vec_fix[5]
    p_only3_b <- p_only3_b+vec_fix[6]
    p_4a <- p_4a+vec_fix[7]
    p_4b <- p_4b+vec_fix[8]
  } #fixing probs, "giving back" probability taken for seeing both on same round 
  
  #format: HC, p_a, p_b, p_o, p_only_2a_2b, p_a_and_o, p_b_and_o, p_only3_a, p_only3_b, p_3o, p_fh_3a, p_fh_3b, p_4a, p_4b
  list((p_na_and_nb - p_only2_o -p_3o),c(p_only2_a,p_only2_b,p_only2_o),c(p_a_and_o, p_b_and_o, p_only_2a_2b),c(p_only3_a, p_only3_b, p_3o),0,0,c(p_fh_3a_2b, p_fh_3b_2a),c(p_4a, p_4b),0)
}

pairingOdds_2equal <- function(cr_a,n,others_remaining)
{
  frcs <- ifelse(((n-2) %/% 10 < 3),ceiling((((n-2)%%10 + 1)%/%2)*0.5 + 0.5),3)
  others_probs <- ((2:4)*others_remaining[2:4])/sum((1:4)*others_remaining)
  
  p_na <- phyper(0,cr_a,n-cr_a,frcs+6) # no pair for a
  p_fh_3o_2a <- (p_na)*
    (others_probs[2]*(phyper(1,2,n- cr_a - 2,3)-phyper(0,2,n- cr_a - 2,3))*(phyper(1,1,n-2- cr_a - 4,3)-phyper(0,1,n-2- cr_a - 4,3))+
       others_probs[3]*(phyper(2,3,n- cr_a - 2,3)-phyper(1,3,n- cr_a - 2,3))*(phyper(1,1,n-1- cr_a - 3,3)-phyper(0,1,n-1- cr_a - 3,3))+
       others_probs[3]*(phyper(1,3,n- cr_a - 2,3)-phyper(0,3,n- cr_a - 2,3))*(1-phyper(0,2,n-2- cr_a - 4,3)))
  p_only2_a_and_o <- (p_na)*(sum(others_probs*(1-phyper(0,1:3,n- cr_a - 1:3,6))) + sum(others_probs*(1-phyper(0,1:3,n- cr_a - 1:3,3))) - 2*p_fh_3o_2a)
  if (cr_a == 2)
  {
    p_4a <- (phyper(1,2,n-2,frcs) - phyper(0,2,n-2,frcs))*(phyper(1,1,n-2-frcs+1,3) - phyper(0,1,n-2-frcs+1,3))+ #11x
      phyper(0,2,n-2,frcs)*(phyper(1,2,n-2-frcs,3) - phyper(0,2,n-2-frcs,3))*(1 - phyper(0,1,n-4-frcs,3))+ # 011
      (phyper(1,2,n-2,frcs) - phyper(0,2,n-2,frcs))*phyper(0,1,n-2-frcs+1,3)*(1 - phyper(0,1,n-5-frcs+1,3)) # 101
  }
  else
  {
    p_4a <- 0
  }
  p_3a <- (1 - p_na)
  p_fh_3a_2o <- (p_3a-p_4a)*(sum(others_probs*(1-phyper(0,1:3,n- cr_a - 1:3,3))))
  p_only3_a <- p_3a - p_4a - p_fh_3a_2o
  
  list(0,(p_na - p_only2_a_and_o - p_fh_3o_2a),(p_only2_a_and_o),(p_only3_a),0,0,c(p_fh_3a_2o, p_fh_3o_2a),(p_4a),0)
}

foo123 <- function(m)
{
  print(m[1,1])
}

getPlayerOdds2 <- function(player,amounts_per_value,amounts_per_suit,deck)
{
  first_round_cards <- ((length(deck) - 30) %/% 8) + 1 # rounding is done downwards here since opp won't necesarily choose to pair
  amounts_remaining <- c(sum(amounts_per_value==1),sum(amounts_per_value==2),sum(amounts_per_value==3),sum(amounts_per_value==4))
  n_deck <- length(deck)
  
  odds <- list(0,0,0,0,0)
  for(i in 1:5)
  {
    if(player[2,i] == -1)
    {
      val_at_hand <- (player[1,i] %% 13) + 1
      pair_odds <- 1 - phyper(0,amounts_per_value[val_at_hand],n_deck - amounts_per_value[val_at_hand],first_round_cards)
      if(amounts_per_value[val_at_hand]-1 < 2)
      {
        unequal_second_card_odds <- amounts_remaining/sum(amounts_remaining)
        p_pair <- pair_odds*pairingOdds_2equal(amounts_per_value[val_at_hand]-1,n_deck,amounts_remaining)
        p_pair <- p_pair + (1 - pair_odds)*
          (unequal_second_card_odds[1]*pairingOdds_2unequal(amounts_per_value[val_at_hand],1,n_deck,amounts_remaining)+
           unequal_second_card_odds[2]*pairingOdds_2unequal(amounts_per_value[val_at_hand],2,n_deck,amounts_remaining)+
           unequal_second_card_odds[3]*pairingOdds_2unequal(amounts_per_value[val_at_hand],3,n_deck,amounts_remaining))
      }
      else
      {
        unequal_second_card_odds <- (amounts_remaining-c(0,1,0,0))/sum((amounts_remaining-c(0,1,0,0)))
        p_pair <- pair_odds*pairingOdds_2equal(amounts_per_value[val_at_hand]-1,n_deck,amounts_remaining-c(0,1,0,0))
        p_pair <- p_pair + (1 - pair_odds)*
          (unequal_second_card_odds[1]*pairingOdds_2unequal(amounts_per_value[val_at_hand],1,n_deck,amounts_remaining)+
           unequal_second_card_odds[2]*pairingOdds_2unequal(amounts_per_value[val_at_hand],2,n_deck,amounts_remaining)+
           unequal_second_card_odds[3]*pairingOdds_2unequal(amounts_per_value[val_at_hand],3,n_deck,amounts_remaining))
      }
      p_flush <- flushOdds(1,amounts_per_suit[(player[1,i] %/% 13) + 1],n_deck)
      amnts <- c(rep(0,max(5-val_at_hand1,0)),amounts_per_value[max(val_at_hand1-4,1):min(val_at_hand1+4,13)],rep(0,max(val_at_hand1-9,0)))
      p_straight <-  straightOdds(amounts =  amnts,n = n_deck)
      if(val_at_hand1 == 13 && val_at_hand2 %in% 1:4)
        p_straight <- p_straight + straightOdds(amounts = c(rep(0,5),amounts_remaining[1:4]),n = n_deck)
    }
    else
    {
      if((player[1,i] %% 13) == (player[2,i] %% 13))
      {
        val_at_hand <- (player[1,i] %% 13) + 1
        if(amounts_per_value[val_at_hand] < 2)
        {
          p_pair <- pairingOdds_2equal(amounts_per_value[val_at_hand],n_deck,amounts_remaining)
        }
        else
        {
          p_pair <- pairingOdds_2equal(amounts_per_value[val_at_hand],n_deck,amounts_remaining-c(0,1,0,0))
        }
        p_flush <- 0
        p_straight <- 0
      }
      else
      {
        val_at_hand1 <- (player[1,i] %% 13) + 1
        val_at_hand2 <- (player[2,i] %% 13) + 1
        
        ar <- amounts_remaining
        if(amounts_per_value[val_at_hand1]>1)
          ar[amounts_per_value[val_at_hand1]-1] <- ar[amounts_per_value[val_at_hand1]-1]-1
        if(amounts_per_value[val_at_hand2]>1)
          ar[amounts_per_value[val_at_hand2]-1] <- ar[amounts_per_value[val_at_hand2]-1]-1
        
        3
        p_pair <- pairingOdds_2unequal(amounts_per_value[val_at_hand1],amounts_per_value[val_at_hand2],n_deck,ar)
        
        if((player[1,i] %/% 13) == (player[2,i] %/% 13))
        {
          p_flush <- flushOdds(2,amounts_per_suit[(player[1,i] %/% 13) + 1],n_deck)
        }
        else
        {
          p_flush <- 0
        }
        
        additional <- val_at_hand2 - val_at_hand1
        amnts <- c(rep(0,max(5-val_at_hand1,0)),amounts_per_value[max(val_at_hand1-4,1):min(val_at_hand1+4,13)],rep(0,max(val_at_hand1-9,0)))
        3
        p_straight <-  straightOdds(1000,additional,amnts,n_deck)
        if(val_at_hand1 == 13 && val_at_hand2 %in% 1:4)
          p_straight <- p_straight + straightOdds(1000,val_at_hand2,c(rep(0,5),amounts_remaining[1:4]),n_deck)
      }
      
    }
    
    p_f_and_s <- c(0,0,0,0,p_straight,p_flush,0,0,0)
    odds[[i]] <- (p_pair + p_f_and_s)/sum(p_pair + p_f_and_s)
  }
  return(odds)
}

decision <- function(player,opponent,card,deck)
{
  amounts_per_value <- numeric(13)
  amounts_per_suit <- numeric(4)
  for(i in 1:13)
  {
    amounts_per_value[i] <-  sum(deck %% 13 == (i-1))
  }
  for(i in 1:4)
  {
    amounts_per_suit[i] <-  sum(deck %/% 13 == (i-1))
  }
  oppOdds <- 0
  if(((length(deck) - 2) %/% 10) == 3)# if it's round 1
  {
    oppOdds <- getPlayerOdds2(opponent,amounts_per_value,amounts_per_suit,deck)
  }
  return(oppOdds)
  # matrices will be in 0-51 format, think of an elegant way to extract that into info that can be passed to funcs above.
  # get the probs (as shown below for p) for all opponent columns.
  # get the probs (as shown below for p) for each player column as is and for each one if the card was to be inserted,
  # calculate the winning probability (as shown below for temp) for each option and submit the decision as int from 0-4
}

{# p <- 0
# for(i in 2:9)
# {
#   p <- p + a[i]*(sum(b[1:i-1]) + b[i]/2)
# }
# 
# temp[1]*temp[2]*temp[3]*(1-temp[4])*(1-temp[5])+
#   temp[1]*temp[2]*temp[4]*(1-temp[3])*(1-temp[5])+
#   temp[1]*temp[2]*temp[5]*(1-temp[3])*(1-temp[4])+
#   temp[1]*temp[3]*temp[4]*(1-temp[2])*(1-temp[5])+
#   temp[1]*temp[3]*temp[5]*(1-temp[2])*(1-temp[4])+
#   temp[1]*temp[4]*temp[5]*(1-temp[2])*(1-temp[3])+
#   temp[2]*temp[3]*temp[4]*(1-temp[1])*(1-temp[5])+
#   temp[2]*temp[3]*temp[5]*(1-temp[1])*(1-temp[4])+
#   temp[2]*temp[4]*temp[5]*(1-temp[1])*(1-temp[3])+
#   temp[3]*temp[4]*temp[5]*(1-temp[1])*(1-temp[2])+
#   
#   temp[1]*temp[2]*temp[3]*temp[4]*(1-temp[5])+
#   temp[1]*temp[2]*temp[3]*temp[5]*(1-temp[4])+
#   temp[1]*temp[2]*temp[5]*temp[4]*(1-temp[3])+
#   temp[1]*temp[5]*temp[3]*temp[4]*(1-temp[2])+
#   temp[5]*temp[2]*temp[3]*temp[4]*(1-temp[1])+
#   
#   temp[1]*temp[2]*temp[3]*temp[4]*temp[5]
}

m <- matrix(rep(-1,25),nrow = 5)
d <- 0:51
s <- sample(d,17)
m[1,] <- s[1:5]
m[2,1:3] <- s[6:8]
c <- s[9]
d <- subset(d,!(d %in% s))

mv <- matrix(rep(-1,25),nrow = 5)
ms <- matrix(rep(-1,25),nrow = 5)
mv[1,] <- s[1:5] %% 13
mv[2,1:3] <- s[6:8] %% 13
ms[1,] <- s[1:5] %/% 13
ms[2,1:3] <- s[6:8] %/% 13

apv <- numeric(13)
aps <- numeric(4)

names(apv) <- 0:12
names(aps) <- 0:3

for(i in 1:13)
{
  apv[i] <-  sum(d %% 13 == (i-1))
}
for(i in 1:4)
{
  aps[i] <-  sum(d %/% 13 == (i-1))
}

getPlayerOdds2(m,apv,aps,d)
mv
ms 
apv
aps

# for HC: check odds that the player with the currently lower HC status can't overcome opp. HC
#(1-phyper(0,sum(apv[h:13]),length(d)-sum(apv[h:13]),9))/2

# last time I changed the output for the pairing functions to a list of 9, containing vec's with the diff probs for each hand
# the idea is to use those for tiebreakers, although a problem arises when applying to 1 card opponent column
# since 2nd card is unknown (possible fix: using avg. value of cards left, per amounts remaining values)