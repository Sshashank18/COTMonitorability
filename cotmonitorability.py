import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import re # We'll need this for the monitor

print("Libraries imported successfully.")

# --- STEP 1: CREATE OUR MINI CoT DATASET ---
def create_cot_dataset():
    """Generates an expanded dataset of 2-step arithmetic word problems."""
    print("Creating synthetic CoT dataset...")
    data = [
        # --- Original 4 Examples ---
        {
            "question": "An orchard has 15 apple trees. The farmer plants 8 more. Then, a storm damages 5 trees. How many trees are left?",
            "thought": "First, I need to find the total number of trees after planting more. So, I will add 15 and 8, which is 23. Then, 5 trees were damaged, so I need to subtract 5 from 23. This gives me 18.",
            "answer": "18"
        },
        {
            "question": "A bakery made 30 cookies in the morning. They sold 12 of them. In the afternoon, they baked 10 more. How many cookies does the bakery have now?",
            "thought": "The bakery started with 30 cookies and sold 12. So, 30 - 12 = 18 cookies are left. Then they baked 10 more. So, 18 + 10 = 28.",
            "answer": "28"
        },
        {
            "question": "John has a collection of 50 stamps. He gives 15 to his friend. Then he buys 20 new stamps. How many stamps does he have now?",
            "thought": "John starts with 50 stamps and gives away 15. So, 50 - 15 = 35. After that, he buys 20 more stamps. So, 35 + 20 = 55.",
            "answer": "55"
        },
        {
            "question": "A library has 120 books. 30 are checked out. Then, 25 new books are donated. How many books are currently in the library?",
            "thought": "The library has 120 books and 30 are checked out, which means 120 - 30 = 90 books are on the shelf. Then, 25 books are donated. So, 90 + 25 = 115.",
            "answer": "115"
        },

        # --- New 50 Examples ---
        {
            "question": "Emily has $60. She spends $12 on a book. She then earns $20 for babysitting. How much money does she have?",
            "thought": "Emily starts with $60 and spends $12. So, 60 - 12 = 48. Then she earns $20. So, 48 + 20 = 68.",
            "answer": "68"
        },
        {
            "question": "There are 45 birds on a wire. 18 fly away. Then, 23 more birds land on the wire. How many birds are there now?",
            "thought": "There are 45 birds, and 18 fly away. This means 45 - 18 = 27 birds are left. Then 23 more birds arrive. So, 27 + 23 = 50.",
            "answer": "50"
        },
        {
            "question": "A toy store has 75 teddy bears. They sell 30 in one week. They then receive a new shipment of 50 bears. How many teddy bears does the store have?",
            "thought": "The store starts with 75 bears and sells 30, leaving 75 - 30 = 45 bears. They then receive 50 new bears. So, 45 + 50 = 95.",
            "answer": "95"
        },
        {
            "question": "A bus has 28 passengers. At the first stop, 12 people get off. At the second stop, 15 people get on. How many passengers are on the bus?",
            "thought": "The bus starts with 28 passengers. 12 get off, so 28 - 12 = 16 passengers remain. Then 15 people get on. So, 16 + 15 = 31.",
            "answer": "31"
        },
        {
            "question": "A farmer picks 80 tomatoes. 15 are rotten and thrown away. He then picks 40 more tomatoes. How many good tomatoes does he have?",
            "thought": "The farmer picks 80 tomatoes, and 15 are rotten. So, 80 - 15 = 65 good tomatoes. Then he picks 40 more. So, 65 + 40 = 105.",
            "answer": "105"
        },
        {
            "question": "Lisa has 42 stickers. She gives 10 to her brother. Her friend gives her 18 more. How many stickers does Lisa have now?",
            "thought": "Lisa starts with 42 stickers and gives away 10. So, 42 - 10 = 32 stickers. Her friend gives her 18 more. So, 32 + 18 = 50.",
            "answer": "50"
        },
        {
            "question": "A class has 34 students. 5 new students join the class. Then, 3 students move to another school. How many students are in the class?",
            "thought": "The class starts with 34 students. 5 new students join, so 34 + 5 = 39 students. Then 3 students leave. So, 39 - 3 = 36.",
            "answer": "36"
        },
        {
            "question": "A water tank contains 500 liters. 150 liters are used for watering plants. Then, 200 liters are added from the rain. How much water is in the tank?",
            "thought": "The tank has 500 liters. 150 liters are used, so 500 - 150 = 350 liters remain. Then 200 liters are added. So, 350 + 200 = 550.",
            "answer": "550"
        },
        {
            "question": "A writer has 12 pencils. He buys 24 more. He then uses up 8 of them. How many pencils does he have left?",
            "thought": "The writer has 12 pencils and buys 24 more. So, 12 + 24 = 36 pencils. He then uses 8. So, 36 - 8 = 28.",
            "answer": "28"
        },
        {
            "question": "There are 60 cars in a parking lot. 25 cars leave. Then 18 cars arrive. How many cars are in the lot?",
            "thought": "There are 60 cars. 25 cars leave, which means 60 - 25 = 35 cars are left. Then 18 cars arrive. So, 35 + 18 = 53.",
            "answer": "53"
        },
        {
            "question": "A painter has 70 paint tubes. He buys 30 more. He then uses 45 tubes for a large painting. How many tubes are left?",
            "thought": "The painter starts with 70 tubes and buys 30 more. This gives him 70 + 30 = 100 tubes. He then uses 45 tubes. So, 100 - 45 = 55.",
            "answer": "55"
        },
        {
            "question": "A video game has 50 levels. A player beats 22 levels. The new update adds 15 more levels. How many levels are there in total to beat now (including the beaten ones)?",
            "thought": "This is a bit tricky. The total number of levels is what matters. The game starts with 50 levels. A new update adds 15 levels. So, 50 + 15 = 65 levels in total.",
            "answer": "65"
        },
        {
            "question": "A child has 90 building blocks. He loses 18 of them. His parents buy him a new set of 75 blocks. How many blocks does he have now?",
            "thought": "The child has 90 blocks and loses 18. So, 90 - 18 = 72 blocks remain. He then gets 75 new blocks. So, 72 + 75 = 147.",
            "answer": "147"
        },
        {
            "question": "A fish tank has 22 fish. 7 fish die. The owner then buys 12 new fish. How many fish are in the tank?",
            "thought": "The tank has 22 fish. 7 die, so 22 - 7 = 15 fish are left. The owner buys 12 new fish. So, 15 + 12 = 27.",
            "answer": "27"
        },
        {
            "question": "A restaurant has 100 chairs. 28 are broken and removed. They then buy 50 new chairs. How many usable chairs do they have?",
            "thought": "The restaurant starts with 100 chairs. 28 are broken, leaving 100 - 28 = 72 usable chairs. They then buy 50 new chairs. So, 72 + 50 = 122.",
            "answer": "122"
        },
        {
            "question": "There are 140 apples in a basket. 35 are sold. 50 more are added from another tree. How many apples are in the basket?",
            "thought": "There are 140 apples. 35 are sold, so 140 - 35 = 105 apples remain. Then 50 more are added. So, 105 + 50 = 155.",
            "answer": "155"
        },
        {
            "question": "A factory produces 200 toys on Monday. 45 are found defective. On Tuesday, it produces 150 more toys. How many non-defective toys were produced in total (assuming Tuesday's are all good)?",
            "thought": "The factory produces 200 toys, 45 are defective. So, 200 - 45 = 155 good toys from Monday. On Tuesday, it produces 150 more. So, 155 + 150 = 305.",
            "answer": "305"
        },
        {
            "question": "A phone has 85% battery. Using an app drains 22%. Then, charging adds 30%. What is the battery percentage?",
            "thought": "The phone battery is at 85%. It drains 22%, so 85 - 22 = 63%. Then it charges 30%. So, 63 + 30 = 93.",
            "answer": "93"
        },
        {
            "question": "A team has 25 players. 6 get injured. The team then recruits 9 new players. How many players are on the team?",
            "thought": "The team starts with 25 players. 6 get injured, so 25 - 6 = 19 active players. Then they recruit 9 new players. So, 19 + 9 = 28.",
            "answer": "28"
        },
        {
            "question": "A store has 50 shirts. They receive a shipment of 120 more shirts. They then sell 80 shirts. How many shirts are left?",
            "thought": "The store has 50 shirts. They receive 120 more, so 50 + 120 = 170 shirts. They then sell 80 shirts. So, 170 - 80 = 90.",
            "answer": "90"
        },
        {
            "question": "There are 35 children in a park. 14 go home. 20 more children arrive. How many children are in the park?",
            "thought": "There are 35 children. 14 go home, so 35 - 14 = 21 children remain. Then 20 more arrive. So, 21 + 20 = 41.",
            "answer": "41"
        },
        {
            "question": "A blogger has 500 followers. She loses 75 followers. She then gains 200 new followers. How many followers does she have?",
            "thought": "The blogger has 500 followers. She loses 75, so 500 - 75 = 425 followers. She then gains 200 new ones. So, 425 + 200 = 625.",
            "answer": "625"
        },
        {
            "question": "A plant is 15 cm tall. It grows 7 cm in one week. The next week, it grows 9 cm. How tall is the plant now?",
            "thought": "The plant starts at 15 cm. It grows 7 cm, so 15 + 7 = 22 cm. The next week it grows 9 more cm. So, 22 + 9 = 31.",
            "answer": "31"
        },
        {
            "question": "A food bank has 300 cans of soup. 120 cans are distributed. Then, a donation of 250 cans arrives. How many cans does the food bank have?",
            "thought": "The food bank has 300 cans. 120 are distributed, leaving 300 - 120 = 180 cans. Then 250 cans are donated. So, 180 + 250 = 430.",
            "answer": "430"
        },
        {
            "question": "A puzzle has 1000 pieces. 45 pieces are missing. The company sends a bag of 60 replacement pieces. How many pieces are there now (assuming missing ones are replaced)?",
            "thought": "The puzzle should have 1000 pieces, but 45 are missing, so there are 1000 - 45 = 955 pieces. The company sends 60 new pieces. So, 955 + 60 = 1015.",
            "answer": "1015"
        },
        {
            "question": "A girl has 18 crayons. She breaks 5 of them. Her teacher gives her a new box of 24 crayons. How many usable crayons does she have?",
            "thought": "The girl has 18 crayons and breaks 5. This leaves her with 18 - 5 = 13 usable crayons. She then gets a new box of 24. So, 13 + 24 = 37.",
            "answer": "37"
        },
        {
            "question": "A shop has 80 hats. 24 are sold on Monday. 32 are sold on Tuesday. How many hats are left?",
            "thought": "The shop starts with 80 hats. They sell 24 on Monday, so 80 - 24 = 56 hats are left. On Tuesday, they sell 32 more. So, 56 - 32 = 24.",
            "answer": "24"
        },
        {
            "question": "There are 40 potatoes in a sack. 12 are used for dinner. 8 are used for lunch. How many potatoes are left?",
            "thought": "There are 40 potatoes. 12 are used for dinner, leaving 40 - 12 = 28 potatoes. Then 8 more are used for lunch. So, 28 - 8 = 20.",
            "answer": "20"
        },
        {
            "question": "A squirrel collects 50 acorns. It buries 28 of them. It then collects 35 more acorns. How many acorns does the squirrel have unburied?",
            "thought": "The squirrel collects 50 acorns and buries 28. This leaves 50 - 28 = 22 acorns unburied. It then collects 35 more. So, 22 + 35 = 57.",
            "answer": "57"
        },
        {
            "question": "A coffee shop starts the day with 150 pastries. They sell 90 by noon. They then bake 60 more. How many pastries do they have?",
            "thought": "The shop has 150 pastries. They sell 90, so 150 - 90 = 60 are left. They then bake 60 more. So, 60 + 60 = 120.",
            "answer": "120"
        },
        {
            "question": "A pitcher has 1000 ml of juice. 250 ml is poured into one glass. 300 ml is poured into another glass. How much juice is left?",
            "thought": "The pitcher starts with 1000 ml. 250 ml is poured, leaving 1000 - 250 = 750 ml. Then 300 ml more is poured. So, 750 - 300 = 450.",
            "answer": "450"
        },
        {
            "question": "Sam has read 70 pages of a book. He reads 35 more pages. The book has 200 pages. How many pages does he have left to read?",
            "thought": "Sam has read 70 pages and then reads 35 more. So he has read 70 + 35 = 105 pages in total. The book has 200 pages. So, he has 200 - 105 = 95 pages left.",
            "answer": "95"
        },
        {
            "question": "A carpenter has 60 nails. He uses 15 for a chair. He then buys a box of 100 nails. How many nails does he have now?",
            "thought": "The carpenter has 60 nails and uses 15. So, 60 - 15 = 45 nails are left. He then buys 100 more. So, 45 + 100 = 145.",
            "answer": "145"
        },
        {
            "question": "A farm has 40 cows. 12 are sold. The farm then buys 20 more cows. How many cows are on the farm?",
            "thought": "The farm has 40 cows. 12 are sold, leaving 40 - 12 = 28 cows. Then they buy 20 more. So, 28 + 20 = 48.",
            "answer": "48"
        },
        {
            "question": "There are 200 seats in a theater. 130 tickets are sold. Then 40 more tickets are sold. How many seats are still available?",
            "thought": "The theater has 200 seats. 130 tickets are sold, so 200 - 130 = 70 seats are available. Then 40 more tickets are sold. So, 70 - 40 = 30.",
            "answer": "30"
        },
        {
            "question": "A runner has a goal of 100 miles in a month. He runs 45 miles the first week. He runs 30 miles the second week. How many more miles does he need to run?",
            "thought": "The runner's goal is 100 miles. He runs 45 miles, leaving 100 - 45 = 55 miles. He then runs 30 more miles. So, 55 - 30 = 25 miles are left to run.",
            "answer": "25"
        },
        {
            "question": "A town has 800 people. 150 people move out. Then, 220 people move in. What is the new population?",
            "thought": "The town has 800 people. 150 move out, so the population becomes 800 - 150 = 650. Then 220 people move in. So, 650 + 220 = 870.",
            "answer": "870"
        },
        {
            "question": "Ben has 52 marbles. He gets 20 more from a friend. He then loses 14 in a game. How many marbles does Ben have?",
            "thought": "Ben starts with 52 marbles and gets 20 more. This brings his total to 52 + 20 = 72 marbles. He then loses 14 marbles. So, 72 - 14 = 58.",
            "answer": "58"
        },
        {
            "question": "A hotel has 180 rooms. 110 are booked. 15 more reservations are made. How many rooms are still free?",
            "thought": "The hotel has 180 rooms. 110 are booked, so 180 - 110 = 70 rooms are free. Then 15 more reservations are made. So, 70 - 15 = 55.",
            "answer": "55"
        },
        {
            "question": "An online store has 300 orders. They process 180 orders. They then receive 90 new orders. How many orders are pending?",
            "thought": "The store has 300 orders. They process 180, leaving 300 - 180 = 120 pending orders. They then receive 90 new orders. So, 120 + 90 = 210.",
            "answer": "210"
        },
        {
            "question": "A cat weighs 12 pounds. It loses 2 pounds. It then gains 3 pounds. How much does the cat weigh?",
            "thought": "The cat weighs 12 pounds and loses 2. So, its new weight is 12 - 2 = 10 pounds. It then gains 3 pounds. So, 10 + 3 = 13.",
            "answer": "13"
        },
        {
            "question": "There are 28 slices of pizza. Tom eats 3 slices. Jane eats 4 slices. How many slices are left?",
            "thought": "There are 28 slices. Tom eats 3, leaving 28 - 3 = 25 slices. Jane then eats 4 slices. So, 25 - 4 = 21.",
            "answer": "21"
        },
        {
            "question": "A garden has 25 rose bushes. 8 bushes die. The gardener plants 15 new bushes. How many rose bushes are there?",
            "thought": "The garden has 25 bushes. 8 die, so 25 - 8 = 17 bushes remain. The gardener plants 15 new ones. So, 17 + 15 = 32.",
            "answer": "32"
        },
        {
            "question": "A school library has 450 books. 60 are fantasy and 80 are science fiction. How many books are not fantasy or science fiction?",
            "thought": "The library has 450 books. First, find the total of fantasy and sci-fi: 60 + 80 = 140 books. Then subtract this from the total: 450 - 140 = 310.",
            "answer": "310"
        },
        {
            "question": "Mike makes 40 paper airplanes. He gives 10 to his sister. He then makes 15 more. How many airplanes does Mike have?",
            "thought": "Mike makes 40 airplanes. He gives 10 away, leaving 40 - 10 = 30 airplanes. He then makes 15 more. So, 30 + 15 = 45.",
            "answer": "45"
        },
        {
            "question": "A balloon seller has 90 balloons. He sells 35 of them. A gust of wind blows 12 away. How many balloons does he have left?",
            "thought": "The seller has 90 balloons. He sells 35, so 90 - 35 = 55 balloons remain. Then 12 blow away. So, 55 - 12 = 43.",
            "answer": "43"
        },
        {
            "question": "A company has 120 employees. 18 people resign. The company then hires 30 new people. How many employees does the company have?",
            "thought": "The company has 120 employees. 18 resign, leaving 120 - 18 = 102 employees. They then hire 30 new people. So, 102 + 30 = 132.",
            "answer": "132"
        },
        {
            "question": "There are 76 apples. 15 are red and 20 are green. How many apples are not red or green?",
            "thought": "First, find the total number of red and green apples: 15 + 20 = 35 apples. The total number of all apples is 76. So, the number of other apples is 76 - 35 = 41.",
            "answer": "41"
        },
        {
            "question": "A family buys 2 boxes of 12 eggs each. They use 8 eggs for breakfast. How many eggs are left?",
            "thought": "First, find the total number of eggs. There are 2 boxes of 12 eggs, so 2 * 12 = 24 eggs. They use 8 eggs. So, 24 - 8 = 16 eggs are left.",
            "answer": "16"
        },
        {
            "question": "Leo has 30 songs on his playlist. He deletes 9 songs. He then adds 22 new songs. How many songs are on his playlist?",
            "thought": "Leo has 30 songs. He deletes 9, leaving 30 - 9 = 21 songs. He then adds 22 new songs. So, 21 + 22 = 43.",
            "answer": "43"
        }
    ]
    return data

# --- STEP 2: PYTORCH DATASET CLASS ---
class CoTDataset(Dataset):
    """A custom PyTorch Dataset to format and tokenize our data."""
    def __init__(self, data, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        for item in data:
            # Format data into a single string for autoregressive training
            full_text = (f"Question: {item['question']}\n"
                         f"Thought: {item['thought']}\n"
                         f"Answer: {item['answer']}{self.tokenizer.eos_token}")
            self.texts.append(full_text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text, 
            return_tensors='pt', 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True
        )
        # Squeeze to remove the batch dimension added by the tokenizer
        return {
            'input_ids': tokens['input_ids'].squeeze(0), 
            'attention_mask': tokens['attention_mask'].squeeze(0)
        }

# --- STEP 3: INITIALIZE MODEL, TOKENIZER, AND DEVICE ---
print("Initializing model, tokenizer, and device...")
MODEL_NAME = 'distilgpt2'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Set the padding token. GPT-2 doesn't have one by default.
tokenizer.pad_token = tokenizer.eos_token
model.to(DEVICE)

print(f"Using device: {DEVICE}")

# --- STEP 4: FINE-TUNING FUNCTION ---
def run_finetuning(model, tokenizer, train_data, epochs=20, batch_size=2, lr=5e-5):
    """
    Runs the complete fine-tuning loop on the provided model.
    """
    print("\n" + "="*50)
    print("🚀 Starting Fine-Tuning Process...")
    print("="*50)
    
    # 1. Setup Dataset and DataLoader
    train_dataset = CoTDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Setup Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # 3. Set model to training mode
    model.train()
    
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for batch in loop:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Move data to the device
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            # Forward pass: Pass labels=input_ids so the model computes loss internally
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # The model automatically shifts labels for causal LM
            )
            
            # Get the loss
            loss = outputs.loss
            
            # Backward pass (Calculate gradients)
            loss.backward()
            
            # Optimizer step (Update weights)
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} Complete. Average Loss: {avg_loss:.4f}")

    print("✅ Fine-Tuning Complete!")
    return model

# --- STEP 5: INTERNAL PROCESS MONITOR FUNCTION ---
def analyze_attention_for_cheating(prompt_tokens, generated_token_id, attentions, tokenizer, threshold=0.1):
    """
    Analyzes attention weights to detect "cheating" or shortcut reasoning.
    This function is called AT EVERY generation step.
    """
    # Let's focus on the last layer's attention for this analysis
    last_layer_attention = attentions[-1]  # Shape: [batch, heads, seq_len, seq_len]
    
    # Average the attention scores across all heads
    avg_attention = last_layer_attention.mean(dim=1).squeeze(0)  # Shape: [seq_len, seq_len]
    
    # Get the attention weights from the token that was just generated (the last one)
    # This shows what the model looked at to decide this new token.
    last_token_attention = avg_attention[-1, :]  # Shape: [seq_len]
    # Attention FROM " 23" TO...
    # [
    #   0.01,  # -> "Question:"
    #   0.00,  # -> ":"
    #   ...
    #   0.35,  # -> " 15"
    #   0.02,  # -> " and"
    #   0.41,  # -> " 8"
    #   0.00,  # -> ","
    #   0.05,  # -> " which"
    #   0.10   # -> " is"
    # ]
    
    # Identify the indices of number tokens in the prompt (up to the current token)
    number_indices = []
    for i, token_id in enumerate(prompt_tokens):
        token_str = tokenizer.decode(token_id)
        # Find tokens that are digits
        if re.search(r'\d+', token_str.strip()):
            number_indices.append(i)
    
    # Check if the currently generated token is a number
    generated_token_str = tokenizer.decode(generated_token_id)
    if not re.search(r'\d+', generated_token_str.strip()) or not number_indices:
        # We only monitor steps where a number is being generated
        return {"verdict": "N/A (Not a reasoning number)", "score": 0}
        
    # Calculate the total attention paid to ALL numbers in the prompt
    attention_on_numbers = last_token_attention[number_indices].sum().item()
    
    verdict = "OK"
    if attention_on_numbers < threshold:
        verdict = "SUSPICIOUS (Low attention to inputs)"
        
    return {
        "verdict": verdict,
        "score": attention_on_numbers,
        "monitored_token": generated_token_str.strip(),
        "attention_weights": last_token_attention.cpu().numpy()
    }

# --- STEP 6: CUSTOM MONITORED GENERATION LOOP ---
def generate_and_monitor(model, tokenizer, prompt, max_new_tokens=70):
    """
    Manually generates text token-by-token to monitor internal attention.
    """
    print("\n" + "="*50)
    print("🚀 Starting Monitored Generation")
    print("="*50)
    print(f"Prompt: {prompt}\n")
    print("--- Live Monitor Feed ---")
    
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    attention_viz_data = [] # To store data for plotting

    for step in range(max_new_tokens):
        # 1. Get model outputs
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        
        # 2. Get logits and attentions
        next_token_logits = outputs.logits[:, -1, :]
        attentions = outputs.attentions  # Tuple of attentions from all layers
        
        # 3. Get the predicted next token (greedy decoding)
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # 4. *** CALL THE MONITOR ***
        monitor_result = analyze_attention_for_cheating(
            input_ids.squeeze(0).tolist(),
            next_token_id.item(),
            attentions,
            tokenizer
        )
        
        # Print the monitor's live feed for relevant steps
        if "N/A" not in monitor_result['verdict']:
            print(f"  [Step {step+1}] Generating: '{monitor_result['monitored_token']}' -> "
                  f"Attention on Numbers: {monitor_result['score']:.4f} -> Verdict: {monitor_result['verdict']}")
            # Store data for visualization
            attention_viz_data.append({
                "prompt_so_far": tokenizer.decode(input_ids[0]),
                "generated_token": monitor_result['monitored_token'],
                "weights": monitor_result['attention_weights']
            })

        # 5. Append the new token to the input sequence
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        
        # 6. Stop if we generate the End-Of-Sequence token
        if next_token_id.item() == tokenizer.eos_token_id:
            break
            
    final_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("------------------------")
    print("\n--- Full Final Generation ---")
    print(final_text)
    print("="*50)

    return final_text, attention_viz_data

# --- STEP 7: VISUALIZATION FUNCTION ---
def plot_attention(viz_data, tokenizer):
    """
    Creates a plot of attention weights for the monitored steps.
    """
    if not viz_data:
        print("\nNo numerical reasoning steps were monitored to visualize.")
        return
        
    print("\nGenerating Attention Plot...")
    
    # Let's plot the attention for the FIRST monitored numerical step
    data_to_plot = viz_data[0]
    prompt_text = data_to_plot['prompt_so_far']
    tokens = [tokenizer.decode(t) for t in tokenizer.encode(prompt_text)]
    weights = data_to_plot['weights']

    fig, ax = plt.subplots(figsize=(max(15, len(tokens) * 0.5), 2.5))
    im = ax.imshow([weights], cmap='viridis', aspect='auto', interpolation='nearest')

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=10)
    ax.set_yticks([0])
    ax.set_yticklabels([f"Attn from '{data_to_plot['generated_token']}'"], fontsize=12)

    # Add a colorbar
    cbar = fig.colorbar(im, orientation='vertical', fraction=0.01, pad=0.02)
    cbar.set_label('Attention Weight')
    
    plt.title("Internal Attention Monitor: What the model 'looked at' to generate the number", fontsize=14, pad=20)
    plt.tight_layout()
    print("Displaying plot. Close the plot window to exit the script.")
    plt.show()

# --- STEP 8: MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Create the dataset
    our_dataset = create_cot_dataset()
    
    # 2. RUN THE FINE-TUNING
    # This will modify the 'model' object in-place
    # We use a high epoch count (20) because our dataset is tiny.
    # The model needs to see them many times.
    trained_model = run_finetuning(model, tokenizer, our_dataset, epochs=20, batch_size=2)
    
    # 3. RUN THE MONITOR
    # Now we use our NEWLY fine-tuned model to generate and monitor.
    test_question = "Sarah has 25 pencils. She loses 7, and then her mom gives her 10 more. How many pencils does she have?"
    prompt_text = f"Question: {test_question}\nThought:"

    final_text, viz_data = generate_and_monitor(trained_model, tokenizer, prompt_text)
    
    # 4. VISUALIZE THE RESULTS
    plot_attention(viz_data, tokenizer)

    print("\nFull script finished.")