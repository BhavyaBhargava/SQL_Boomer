import sqlite3
import random
from datetime import datetime, timedelta

# Create or connect to the SQLite database
conn = sqlite3.connect('enterprise_mock.db')
cursor = conn.cursor()

print("🔨 Rebuilding Enterprise Mock Database...")

# 1. Create Tables
cursor.executescript('''
    DROP TABLE IF EXISTS tblMilestones;
    DROP TABLE IF EXISTS tblHolds;
    DROP TABLE IF EXISTS tblSales;
    DROP TABLE IF EXISTS tblLots;
    DROP TABLE IF EXISTS tblCommunities;
    DROP TABLE IF EXISTS tblBuilders;

    CREATE TABLE tblCommunities (
        CommunityID INTEGER PRIMARY KEY,
        CommunityName VARCHAR(100),
        Region VARCHAR(50),
        TotalLots INTEGER
    );

    CREATE TABLE tblBuilders (
        BuilderID INTEGER PRIMARY KEY,
        BuilderName VARCHAR(100),
        BuilderRating VARCHAR(10),
        ActiveStatus INTEGER
    );

    CREATE TABLE tblLots (
        LotID INTEGER PRIMARY KEY,
        CommunityIDfk INTEGER,
        BuilderIDfk INTEGER,
        LotNumber VARCHAR(50),
        Status VARCHAR(50),
        BasePrice DECIMAL(10, 2),
        IsSpec INTEGER,
        FOREIGN KEY(CommunityIDfk) REFERENCES tblCommunities(CommunityID),
        FOREIGN KEY(BuilderIDfk) REFERENCES tblBuilders(BuilderID)
    );

    CREATE TABLE tblSales (
        SaleID INTEGER PRIMARY KEY,
        LotIDfk INTEGER,
        SaleDate DATETIME,
        SalePrice DECIMAL(10, 2),
        BuyerName VARCHAR(100),
        ClosingDate DATETIME,
        FOREIGN KEY(LotIDfk) REFERENCES tblLots(LotID)
    );

    CREATE TABLE tblHolds (
        HoldID INTEGER PRIMARY KEY,
        LotIDfk INTEGER,
        HoldReason VARCHAR(50),
        StartDate DATETIME,
        IsResolved INTEGER,
        FOREIGN KEY(LotIDfk) REFERENCES tblLots(LotID)
    );

    CREATE TABLE tblMilestones (
        MilestoneID INTEGER PRIMARY KEY,
        LotIDfk INTEGER,
        TrenchDate DATETIME,
        FramingDate DATETIME,
        CompletionDate DATETIME,
        FOREIGN KEY(LotIDfk) REFERENCES tblLots(LotID)
    );
''')

# ==========================================
# 2. GENERATE MASSIVE DUMMY DATASET
# ==========================================
print("🧬 Generating 500 lots of mock data...")

now = datetime.now()
random.seed(42) # For consistent demo data

# --- COMMUNITIES (15) ---
regions = ['East', 'West', 'North', 'South', 'Central']
community_names = [
    'Whispering Pines', 'Desert Mirage', 'Maplewood Estates', 'Oak Hollow', 'Sunland Springs',
    'Riverwalk', 'Mountain View', 'Pebble Creek', 'Silver Lake', 'Cedar Ridge',
    'Highland Park', 'Valley Forge', 'Stonecrest', 'Ironwood', 'Copper Canyons'
]
communities = []
for i in range(1, 16):
    communities.append((i, community_names[i-1], random.choice(regions), random.randint(50, 300)))

# --- BUILDERS (10) ---
builder_names = [
    'Apex Construction', 'Value Homes LLC', 'Pinnacle West', 'Lennar', 'Toll Brothers',
    'DR Horton', 'Pulte', 'Taylor Morrison', 'KB Home', 'Meritage Homes'
]
ratings = ['A+', 'A', 'B', 'C']
builders = []
for i in range(1, 11):
    builders.append((i, builder_names[i-1], random.choice(ratings), 1))

# --- LOTS, MILESTONES, SALES, HOLDS (500) ---
lots = []
milestones = []
sales = []
holds = []

statuses = ['Pre-Construction', 'Permit', 'Framing', 'Drywall', 'Finishes', 'Complete', 'Closed', 'Cancelled']
first_names = ['John', 'Jane', 'Michael', 'Emily', 'David', 'Sarah', 'Chris', 'Amanda', 'Robert', 'Jessica']
last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']

sale_id_counter = 1
hold_id_counter = 1

for lot_id in range(1, 501):
    community_id = random.randint(1, 15)
    builder_id = random.randint(1, 10)
    lot_number = f"L-{1000 + lot_id}"
    status = random.choice(statuses)
    
    # Base price between 300k and 1.2M (to trigger "Big Ticket" Glossary item)
    base_price = round(random.uniform(300000, 1200000), 2)
    is_spec = random.choices([0, 1], weights=[0.4, 0.6])[0] # 60% chance it's a Spec home
    
    lots.append((lot_id, community_id, builder_id, lot_number, status, base_price, is_spec))

    # Calculate realistic dates based on status
    trench_dt, framing_dt, complete_dt = None, None, None
    days_ago_start = random.randint(30, 300)
    
    if status in ['Framing', 'Drywall', 'Finishes', 'Complete', 'Closed']:
        trench_dt = (now - timedelta(days=days_ago_start)).strftime('%Y-%m-%d %H:%M:%S')
    if status in ['Drywall', 'Finishes', 'Complete', 'Closed']:
        framing_dt = (now - timedelta(days=days_ago_start - 30)).strftime('%Y-%m-%d %H:%M:%S')
    if status in ['Complete', 'Closed']:
        # Force some "Stale Inventory" by making completion date > 90 days ago
        if is_spec == 1 and status == 'Complete' and random.random() < 0.3:
            comp_days_ago = random.randint(95, 150)
        else:
            comp_days_ago = random.randint(5, 80)
        complete_dt = (now - timedelta(days=comp_days_ago)).strftime('%Y-%m-%d %H:%M:%S')

    milestones.append((lot_id, lot_id, trench_dt, framing_dt, complete_dt))

    # Generates Sales Data
    # If not a spec, or if it's closed/closing soon, it needs a sale record
    if is_spec == 0 or status == 'Closed' or (status == 'Complete' and random.random() < 0.5):
        buyer = f"{random.choice(first_names)} {random.choice(last_names)}"
        sale_price = base_price + random.uniform(10000, 75000) # Add upgrade premiums
        
        sale_date = now - timedelta(days=random.randint(40, 300))
        
        # Determine Closing Date
        if status == 'Closed':
            closing_date = sale_date + timedelta(days=random.randint(30, 60))
        elif status == 'Cancelled':
            closing_date = None # Never closed
        else:
            # "Closing Soon" logic (within next 30 days)
            closing_date = now + timedelta(days=random.randint(5, 45))

        sale_date_str = sale_date.strftime('%Y-%m-%d %H:%M:%S')
        closing_date_str = closing_date.strftime('%Y-%m-%d %H:%M:%S') if closing_date else None

        sales.append((sale_id_counter, lot_id, sale_date_str, sale_price, buyer, closing_date_str))
        sale_id_counter += 1

    # Generate Holds
    # 15% chance a lot has/had a hold
    if random.random() < 0.15:
        reasons = ['Schedule', 'Permit', 'Material', 'Buyer']
        reason = random.choice(reasons)
        is_resolved = 1 if status in ['Complete', 'Closed'] else random.choices([0, 1], weights=[0.7, 0.3])[0]
        start_date = (now - timedelta(days=random.randint(10, 100))).strftime('%Y-%m-%d %H:%M:%S')
        
        holds.append((hold_id_counter, lot_id, reason, start_date, is_resolved))
        hold_id_counter += 1


# ==========================================
# 3. INSERT DATA
# ==========================================
cursor.executemany("INSERT INTO tblCommunities VALUES (?, ?, ?, ?)", communities)
cursor.executemany("INSERT INTO tblBuilders VALUES (?, ?, ?, ?)", builders)
cursor.executemany("INSERT INTO tblLots VALUES (?, ?, ?, ?, ?, ?, ?)", lots)
cursor.executemany("INSERT INTO tblMilestones VALUES (?, ?, ?, ?, ?)", milestones)
cursor.executemany("INSERT INTO tblHolds VALUES (?, ?, ?, ?, ?)", holds)
cursor.executemany("INSERT INTO tblSales VALUES (?, ?, ?, ?, ?, ?)", sales)

conn.commit()
conn.close()

print(f"✅ enterprise_mock.db seeded successfully!")
print(f"   - {len(communities)} Communities")
print(f"   - {len(builders)} Builders")
print(f"   - {len(lots)} Lots")
print(f"   - {len(sales)} Sales Transactions")
print(f"   - {len(holds)} Construction Holds")