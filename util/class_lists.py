T2_classes = "panda,hexagon,telephone,mug,stove,laptop,paper clip,crab,birthday cake,squiggle,scorpion,cooler,whale,spider," \
             "spreadsheet,parrot,toaster,airplane,beard,envelope,bandage,snorkel,canoe,sword,ocean,hammer,flying saucer,pizza," \
             "squirrel,kangaroo,basketball,brain,eraser,violin,golf club,screwdriver,potato,sock,floor lamp,swan,sea turtle,moon," \
             "stethoscope,lion,pillow".split(",")

T1_classes = "blackberry,power outlet,peas,hot tub,toothbrush,skateboard,cloud," \
             "elbow,bat,pond,compass,elephant,hurricane,jail,school bus,skyscraper,tornado,picture frame,lollipop,spoon,saw,cup," \
             "roller coaster,pants,jacket,rifle,yoga,toilet,waterslide,axe,snowman,bracelet,basket,anvil,octagon,washing machine,tree," \
             "television,bowtie,sweater,backpack,zebra,suitcase,stairs,The Great Wall of China".split(',')

ST1_classes = "The Eiffel Tower,The Mona Lisa,aircraft carrier,alarm clock,ambulance,angel,animal migration,ant,apple,arm,asparagus," \
              "banana,barn,baseball,baseball bat,bathtub,beach,bear,bed,bee,belt,bench,bicycle,binoculars,bird,blueberry,book,boomerang," \
              "bottlecap,bread,bridge,broccoli,broom,bucket,bulldozer,bus,bush,butterfly,cactus,cake,calculator,calendar,camel,camera," \
              "camouflage,campfire,candle,cannon,car,carrot,castle,cat,ceiling fan,cell phone,cello,chair,chandelier,church,circle," \
              "clarinet,clock,coffee cup,computer,cookie,couch,cow,crayon,crocodile,crown,cruise ship,diamond,dishwasher,diving board," \
              "dog,dolphin,donut,door,dragon,dresser,drill,drums,duck,dumbbell,ear,eye,eyeglasses,face,fan,feather,fence,finger," \
              "fire hydrant,fireplace,firetruck,fish,flamingo,flashlight,flip flops,flower,foot,fork,frog,frying pan,garden,garden hose," \
              "giraffe,goatee,grapes,grass,guitar,hamburger,hand,harp,hat,headphones,hedgehog,helicopter,helmet,hockey puck,hockey stick," \
              "horse,hospital,hot air balloon,hot dog,hourglass,house,house plant,ice cream,key,keyboard,knee,knife,ladder,lantern,leaf," \
              "leg,light bulb,lighter,lighthouse,lightning,line,lipstick,lobster,mailbox,map,marker,matches,megaphone,mermaid,microphone," \
              "microwave,monkey,mosquito,motorbike,mountain,mouse,moustache,mouth,mushroom,nail,necklace,nose,octopus,onion,oven,owl," \
              "paint can,paintbrush,palm tree,parachute,passport,peanut,pear,pencil,penguin,piano,pickup truck,pig,pineapple,pliers," \
              "police car,pool,popsicle,postcard,purse,rabbit,raccoon,radio,rain,rainbow,rake,remote control,rhinoceros,river," \
              "rollerskates,sailboat,sandwich,saxophone,scissors,see saw,shark,sheep,shoe,shorts,shovel,sink,skull,sleeping bag," \
              "smiley face,snail,snake,snowflake,soccer ball,speedboat,square,star,steak,stereo,stitches,stop sign,strawberry," \
              "streetlight,string bean,submarine,sun,swing set,syringe,t-shirt,table,teapot,teddy-bear,tennis racquet,tent,tiger,toe," \
              "tooth,toothpaste,tractor,traffic light,train,triangle,trombone,truck,trumpet,umbrella,underwear,van,vase,watermelon," \
              "wheel,windmill,wine bottle,wine glass,wristwatch,zigzag,blackberry,power outlet,peas,hot tub,toothbrush,skateboard,cloud," \
              "elbow,bat,pond,compass,elephant,hurricane,jail,school bus,skyscraper,tornado,picture frame,lollipop,spoon,saw,cup," \
              "roller coaster,pants,jacket,rifle,yoga,toilet,waterslide,axe,snowman,bracelet,basket,anvil,octagon,washing machine,tree," \
              "television,bowtie,sweater,backpack,zebra,suitcase,stairs,The Great Wall of China".split(',')

ST2_classes = "The Eiffel Tower,The Mona Lisa,aircraft carrier,alarm clock,ambulance,angel,animal migration,ant,apple,arm,asparagus," \
              "banana,barn,baseball,baseball bat,bathtub,beach,bear,bed,bee,belt,bench,bicycle,binoculars,bird,blueberry,book,boomerang," \
              "bottlecap,bread,bridge,broccoli,broom,bucket,bulldozer,bus,bush,butterfly,cactus,cake,calculator,calendar,camel,camera," \
              "camouflage,campfire,candle,cannon,car,carrot,castle,cat,ceiling fan,cell phone,cello,chair,chandelier,church,circle," \
              "clarinet,clock,coffee cup,computer,cookie,couch,cow,crayon,crocodile,crown,cruise ship,diamond,dishwasher,diving board," \
              "dog,dolphin,donut,door,dragon,dresser,drill,drums,duck,dumbbell,ear,eye,eyeglasses,face,fan,feather,fence,finger," \
              "fire hydrant,fireplace,firetruck,fish,flamingo,flashlight,flip flops,flower,foot,fork,frog,frying pan,garden,garden hose," \
              "giraffe,goatee,grapes,grass,guitar,hamburger,hand,harp,hat,headphones,hedgehog,helicopter,helmet,hockey puck,hockey stick," \
              "horse,hospital,hot air balloon,hot dog,hourglass,house,house plant,ice cream,key,keyboard,knee,knife,ladder,lantern,leaf," \
              "leg,light bulb,lighter,lighthouse,lightning,line,lipstick,lobster,mailbox,map,marker,matches,megaphone,mermaid," \
              "microphone,microwave,monkey,mosquito,motorbike,mountain,mouse,moustache,mouth,mushroom,nail,necklace,nose,octopus," \
              "onion,oven,owl,paint can,paintbrush,palm tree,parachute,passport,peanut,pear,pencil,penguin,piano,pickup truck,pig," \
              "pineapple,pliers,police car,pool,popsicle,postcard,purse,rabbit,raccoon,radio,rain,rainbow,rake,remote control,rhinoceros," \
              "river,rollerskates,sailboat,sandwich,saxophone,scissors,see saw,shark,sheep,shoe,shorts,shovel,sink,skull,sleeping bag," \
              "smiley face,snail,snake,snowflake,soccer ball,speedboat,square,star,steak,stereo,stitches,stop sign,strawberry," \
              "streetlight,string bean,submarine,sun,swing set,syringe,t-shirt,table,teapot,teddy-bear,tennis racquet,tent,tiger,toe," \
              "tooth,toothpaste,tractor,traffic light,train,triangle,trombone,truck,trumpet,umbrella,underwear,van,vase,watermelon," \
              "wheel,windmill,wine bottle,wine glass,wristwatch,zigzag,panda,hexagon,telephone,mug,stove,laptop,paper clip,crab," \
              "birthday cake,squiggle,scorpion,cooler,whale,spider,spreadsheet,parrot,toaster,airplane,beard,envelope,bandage,snorkel," \
              "canoe,sword,ocean,hammer,flying saucer,pizza,squirrel,kangaroo,basketball,brain,eraser,violin,golf club,screwdriver," \
              "potato,sock,floor lamp,swan,sea turtle,moon,stethoscope,lion,pillow".split(",")

sketchy_class_id_map = {'airplane': {'n02691156'}, 'alarm clock': {'n02694662'}, 'ant': {'n02219486'},
                        'ape': {'n02480495', 'n02470325', 'n02483708', 'n02483362', 'n02481823', 'n02481500'},
                        'apple': {'n07739125'}, 'armor': {'n03146219', 'n02895154', 'n03000247'}, 'axe': {'n02764044'},
                        'banana': {'n07753592'}, 'bat': {'n02142407', 'n02147947', 'n02149420', 'n02147328', 'n02139199'},
                        'bear': {'n02131653'}, 'bee': {'n02206856'},
                        'beetle': {'n02167151', 'n02169497', 'n02168699', 'n02165105', 'n02176261'}, 'bell': {'n02824448', 'n03028596'},
                        'bench': {'n02828884', 'n03891251'}, 'bicycle': {'n02834778', 'n03792782', 'n04126066'}, 'blimp': {'n02850950'},
                        'bread': {'n07687211', 'n07687469', 'n07682316', 'n07683786', 'n07679356', 'n07684084'}, 'butterfly': {'n02274259'},
                        'cabin': {'n03686924', 'n02932400'}, 'camel': {'n02437136'}, 'candle': {'n02948072'}, 'cannon': {'n02950826'},
                        'car (sedan)': {'n04166281', 'n02958343'}, 'castle': {'n02980441'}, 'cat': {'n02121620'},
                        'chair': {'n02738535', 'n03001627'}, 'chicken': {'n01791625'}, 'church': {'n03028079'}, 'couch': {'n04256520'},
                        'cow': {'n02406174', 'n02404432', 'n02403454', 'n01887787', 'n02404186'},
                        'crab': {'n01979526', 'n01978287', 'n01981276', 'n01980166', 'n01978930', 'n01978455'},
                        'crocodilian': {'n01697457', 'n01698640'}, 'cup': {'n03147509', 'n03063073'},
                        'deer': {'n02433729', 'n02431337', 'n02431628', 'n02431441', 'n02430045', 'n02432983', 'n02433925'},
                        'dog': {'n02106662', 'n02103406', 'n02109525'}, 'dolphin': {'n02068974', 'n02070430', 'n02072040'},
                        'door': {'n03222318', 'n03226880', 'n03222176'}, 'duck': {'n01846331'}, 'elephant': {'n02503517'},
                        'eyeglasses': {'n04272054', 'n04356056'}, 'fan': {'n03271574'},
                        'fish': {'n02567633', 'n02564403', 'n02542958', 'n02539424', 'n02631628', 'n02655020', 'n02607470', 'n02534734', 'n02513560', 'n02641379', 'n02607072', 'n01440764', 'n02512053', 'n02605316', 'n02537525', 'n02643566', 'n02606052', 'n02514041', 'n01447331'},
                        'flower': {'n11978713', 'n11669921', 'n11939491'}, 'frog': {'n01639765', 'n01641577', 'n01641391'},
                        'geyser': {'n09288635'}, 'giraffe': {'n02439033'}, 'guitar': {'n03467517', 'n02676566', 'n03272010'},
                        'hamburger': {'n07697100', 'n07697313'}, 'hammer': {'n03481172'}, 'harp': {'n03495258'},
                        'hat': {'n03124170', 'n03028785', 'n04259630', 'n02859184', 'n03497657', 'n02954340'},
                        'hedgehog': {'n01872401', 'n02346627', 'n01894207'}, 'helicopter': {'n03512147'}, 'hermit crab': {'n01986214'},
                        'horse': {'n02374451'}, 'hot-air balloon': {'n03541923', 'n02782093'}, 'hotdog': {'n07697537'},
                        'hourglass': {'n03544143'}, 'jack-o-lantern': {'n03590841'}, 'jellyfish': {'n01910747'},
                        'kangaroo': {'n01877134'}, 'knife': {'n03624134', 'n02973904', 'n02976123'},
                        'lion': {'n02129165'}, 'lizard': {'n01693334', 'n01683558', 'n01674464', 'n01680478'},
                        'lobster': {'n01983481', 'n01985128'}, 'motorcycle': {'n03790512'}, 'mouse': {'n02330245'},
                        'mushroom': {'n12997919', 'n12998815'}, 'owl': {'n01621127'}, 'parrot': {'n01816887', 'n01819313', 'n01817953', 'n01820546'},
                        'pear': {'n07767847', 'n07768230', 'n12651611'}, 'penguin': {'n02055803', 'n02056570'}, 'piano': {'n03452741'},
                        'pickup truck': {'n03930630'}, 'pig': {'n02395406'}, 'pineapple': {'n07753275'}, 'pistol': {'n03948459', 'n04086273'},
                        'pizza': {'n07873807'}, 'pretzel': {'n07695742'}, 'rabbit': {'n02325366'}, 'raccoon': {'n02508021'},
                        'racket': {'n04039381', 'n04409806', 'n02772700'}, 'ray': {'n01498041', 'n01496331'}, 'rhinoceros': {'n02391994'},
                        'rifle': {'n02907391', 'n04090263', 'n02749479', 'n03416775'},
                        'rocket': {'n03773504', 'n04099429', 'n03799375', 'n04415663'}, 'sailboat': {'n04128499'},
                        'saw': {'n03474779', 'n02770585', 'n04140064'}, 'saxophone': {'n04141076'}, 'scissors': {'n04148054', 'n03044934'},
                        'scorpion': {'n01770393'}, 'sea turtle': {'n01664065', 'n01665541', 'n01663401'}, 'seagull': {'n02041246'},
                        'seal': {'n02077923', 'n02076196'}, 'shark': {'n01483021', 'n01491361', 'n01487506', 'n01494475', 'n01482330', 'n01489920'},
                        'sheep': {'n02412080', 'n02413131', 'n02414290', 'n02411705', 'n02412210'},
                        'shoe': {'n02882894', 'n04120489', 'n04199027', 'n03680355', 'n04546081', 'n04593524'},
                        'skyscraper': {'n04233124'}, 'snail': {'n01944390'}, 'snake': {'n01726692', 'n01748264', 'n01752165', 'n01729977'},
                        'songbird': {'n01527347', 'n01569566', 'n01592084', 'n01537544', 'n01558993', 'n01594787', 'n01531178', 'n01530575', 'n01532829', 'n01534433'},
                        'spider': {'n01772222'}, 'spoon': {'n04350769', 'n04284002', 'n04597913', 'n03633091'}, 'squirrel': {'n02355227'},
                        'starfish': {'n02317335'}, 'strawberry': {'n07745940'}, 'swan': {'n01858845', 'n01860187', 'n01858441'},
                        'sword': {'n03039493', 'n04373894', 'n04147793'}, 'table': {'n04379964', 'n03201208', 'n04379243'},
                        'tank': {'n04389033'}, 'teapot': {'n04398044'}, 'teddy bear': {'n04399382'}, 'tiger': {'n02129604'},
                        'tree': {'n12755727', 'n12753007', 'n12726670', 'n11615026', 'n11608250', 'n12202936', 'n12276872', 'n12261808', 'n12587803', 'n12282933', 'n12523475', 'n12199790', 'n12732966', 'n12273939', 'n12304703', 'n11611561', 'n11660300', 'n12281788', 'n11646344', 'n12269241', 'n12242409', 'n11759853', 'n12593994', 'n12713866', 'n12582231', 'n12305089', 'n11622368', 'n11629047'},
                        'trumpet': {'n03110669'}, 'turtle': {'n01669191'}, 'umbrella': {'n04507155'}, 'violin': {'n02803934', 'n04536866'},
                        'volcano': {'n09472597'}, 'wading bird': {'n02012849', 'n02007558', 'n02002724', 'n02000954', 'n02009912'},
                        'wheelchair': {'n04576002'}, 'windmill': {'n04587559', 'n04587404'},
                        'window': {'n04239333', 'n03395514', 'n03227184', 'n04589593', 'n04587648'}, 'wine bottle': {'n04591713'},
                        'zebra': {'n02391049'}}

sketchy_train_list = ['lion', 'hot-air balloon', 'violin', 'tiger', 'eyeglasses', 'ant', 'mouse', 'jack-o-lantern', 'lobster', 'teddy bear',
                      'teapot', 'helicopter', 'duck', 'wading bird', 'rabbit', 'penguin', 'sheep', 'windmill', 'piano', 'jellyfish',
                      'table', 'fan', 'beetle', 'cabin', 'scorpion', 'scissors', 'banana', 'tank', 'umbrella', 'crocodilian', 'volcano',
                      'knife', 'cup', 'saxophone', 'pistol', 'swan', 'chicken', 'sword', 'seal', 'alarm clock', 'rocket', 'guitar',
                      'bicycle', 'owl', 'squirrel', 'hermit crab', 'horse', 'spoon', 'cow', 'hotdog', 'camel', 'turtle', 'pizza', 'spider',
                      'songbird', 'rifle', 'chair', 'starfish', 'tree', 'airplane', 'bread', 'bench', 'harp', 'seagull', 'blimp', 'apple',
                      'geyser', 'trumpet', 'frog', 'lizard', 'axe', 'sea turtle', 'pretzel', 'snail', 'butterfly', 'bear', 'ray',
                      'wine bottle', 'dog', 'armor', 'elephant', 'raccoon', 'rhinoceros', 'door', 'hat', 'deer', 'snake', 'ape', 'flower',
                      'car (sedan)', 'kangaroo', 'dolphin', 'hamburger', 'castle', 'pineapple', 'saw', 'zebra', 'candle', 'cannon',
                      'racket', 'church', 'fish', 'mushroom', 'strawberry', 'window', 'sailboat', 'hourglass', 'cat', 'shoe', 'hedgehog',
                      'couch', 'giraffe', 'hammer', 'motorcycle', 'shark']

sketchy_val_list = ['parrot', 'pickup truck', 'crab', 'skyscraper', 'bat', 'wheelchair', 'pig', 'pear', 'bee', 'bell']

miniimagenet_test = {'n01930112', 'n02129165', 'n02116738', 'n02871525', 'n02099601', 'n04146614', 'n03775546', 'n03146219', 'n02110341',
                     'n02443484', 'n03127925', 'n07613480', 'n01981276', 'n03272010', 'n04149813', 'n04522168', 'n02110063', 'n03544143',
                     'n02219486', 'n04418357'}

miniimagenet_train = {'n04251144', 'n02457408', 'n02074367', 'n04604644', 'n04296562', 'n03998194', 'n02089867', 'n03337140', 'n03838899',
                      'n02101006', 'n03676483', 'n03400231', 'n03062245', 'n07697537', 'n01749939', 'n04596742', 'n01910747', 'n02108551',
                      'n03347037', 'n02108915', 'n01843383', 'n02108089', 'n06794110', 'n03908618', 'n01532829', 'n02966193', 'n02091831',
                      'n02687172', 'n04389033', 'n02747177', 'n04612504', 'n07747607', 'n01704323', 'n03854065', 'n02113712', 'n03924679',
                      'n04275548', 'n02120079', 'n02823428', 'n13054560', 'n03017168', 'n03476684', 'n02795169', 'n03207743', 'n09246464',
                      'n02111277', 'n04243546', 'n13133613', 'n04443257', 'n02165456', 'n02606052', 'n01770081', 'n03527444', 'n04509417',
                      'n04067472', 'n03047690', 'n04515003', 'n01558993', 'n02105505', 'n04435653', 'n03888605', 'n03220513', 'n04258138',
                      'n07584110'}

miniimagenet_eval = {'n01855672', 'n02971356', 'n02114548', 'n03773504', 'n03584254', 'n03417042', 'n02091244', 'n02174001', 'n03075370',
                     'n03535780', 'n02950826', 'n09256479', 'n02138441', 'n03980874', 'n02981792', 'n03770439'}

