# Bonjour
Hier een uitleg over het algo.

## General:
Het algoritme werkt in de volgende volgorde (iedere functie leg ik precies uit in de volgende sectie):
1. Initialisatie random populatie
2. Evaluatie populatie (evaluate)
3. Selecteren van parents (parent_selection)
4. Maken van de offspring (recombination)
5. Muteren van de offspring (mutate)
6. Killen van 80% van de vorige pop (death_match)
7. Muteren van de overige 20% van de pop, met een 50% kans per individu (mutate)
8. Aan elkaar knopen van de overige 20% van de populatie en de nieuwe offspring (wat dus de andere 80% is)
9. Evaluatie nieuwe populatie (als er voor 4 ronden geen nieuwe beste oplossing is, dan vervangen we een deel van de populatie met nieuwe randoms (scramble_pop))
10. Back to step 3.


## Functies specifiek:

### simulation(env, x)
Kennen we.

### evaluate(pop)
Kennen we.

### parent_selection(pop, fit_pop, rounds)
Er worden vanuit de 'main for-loop' (dus waar het EA zelf zich afspeelt) 20 rondes gedaan voor de parent selection, waaraan een nummer *rounds* aan verbonden zit.
Bij iedere ronde worden de 2 hoogste rank individuals gekozen als parents *minus 2x het rondenummer*, ofwel, de eerste ronde geeft als beste parents rank 99 en 98, de 2e ronde geeft rank 97 en 96 etc., tot aan rank 61 en 60 in ronde 20. (Higher = better uiteraard.)
Tevens worden er iedere ronde 3 volledig random parents uitgekozen van de populatie, om er zo voor te zorgen dat we niet te snel convergence krijgen.
Er worden dus per generatie 20 rondes gehouden, met elke ronde 5 parents.

### recombination(parents)
De 5 parents zorgen samen voor 4 kinderen. Twee van deze kinderen komen van 'Whole Arithmetic Recombination', waarbij ik de dirichlet functie gebruik van numpy om 5 random nummers te kiezen die samen 1 zijn. M.a.w.: we krijgen voor een kind bijvoorbeeld 0.2p1 + 0.1p2 + 0.4p3 + 0.1p4 + 0.2p5. Voor kind 2 wordt de dirichlet functie weer aangeroepen om voor een nieuwe combinatie te zorgen.
De andere 2 kinderen worden gemaakt door n-Point Crossover, met in dit geval n=4. Ik kies hier 4 random nummers tussen 1 en 265 om te kiezen voor de crossover points, en ik shuffle de parents-lijst om ervoor te zorgen dat we niet altijd de kop van p1 krijgen en de staart van p5. De nummers voor de crossover points en de volgorde van parents zijn anders voor beide kinderen (uiteraard). Ik knoop de gen-stukken aan elkaar met de concatenate functie van numpy, die je veel zult zien in het algo.

Na 20 rondes hiervan hebben we dus 80 nieuwe kids, de helft met whole arithmetic crossover, de andere helft met n-point.

### mutate(offspring)
Argument hier is offspring, maar ik gebruik het ook voor de overige populatie na het killen. Anyway, ieder individu van de lijst die je in deze functie gooit heeft een 50% kans om te muteren (zodat niet alles altijd verandert). Als een individu muteert, dan lopen we al zijn weights af en is er per weight een kans (mutation waarde bovenaan script, staat op 0.2) dat het muteert. De mutatie zelf wordt gedaan door een getal te kiezen uit een normaalverdeling met een std van 0.2 (up for debate), zodat we niet te hoge waarden krijgen en we niet eindigen met individuen waar heel veel van de weights gewoon de lower of upper limit zijn (-1, 1). De mutaties zijn dus iets meer genuanceerd dan in het standaard script.

### death_match(pop)
In deze functie killen we een fractie van de populatie. We doen wederom npop/5 rondes (met npop 100 dus 20 rondes), waarin 5 random individuen worden uitgekozen en de beste wint. Omdat het er 5 zijn is de kans dus vrij groot dat er iedere ronde een redelijk goede oplossing overleeft (en de beste overleeft sowieso ofc). Hoeveel we er killen kunnen we mee spelen, maar omdat 5 parents 4 kinderen geven is het handig om dit op 80% te laten, zodat we altijd weer een volle populatie van 100 krijgen. De gekozen 5 individuen per ronde worden allemaal uit de pop gehaald, zodat ze niet nog eens in een match kunnen komen. Belangrijk hierbij is dus dat we altijd een npop kiezen waarbij npop%5=0, anders krijgen we een error. (5 kiezen uit een lijst van 3 kan dus niet.)

### scramble_pop(pop)
De 'fresh blood' functie. Als we na 4 generaties geen nieuwe beste oplossing hebben, gooien we de hele populatie in deze functie. Hier worden de slechtste x% gekozen en al hun weights worden vervangen door random weights. Met welke fractie van de populatie we vervangen en na hoeveel gens deze functie wordt aangeroepen kunnen we vrij spelen zonder dependencies op andere functies.
