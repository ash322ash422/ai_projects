from libraries import *
from faker import Faker
from config import SAVED_DIR, DATA_FILENAME

FILE_PATH = SAVED_DIR / DATA_FILENAME

fake = Faker()

# Function to generate 1,000 records
def generate_and_save_data(num_records=1000):
    data = []
    
    # Define the choices for each categorical field
    dia_choices = [74, 75, 76, 77]
    manf_choices = ['Essilor', 'Crizol', 'Hoya']
    material_choices = ['plastic', 'polycarb']
    operator_choices = ['Ash', 'Bob', 'Cathy', 'Dan']
    quality_choices = ['pass', 'fail']
    
    
    # Generate all possible values for SPH and CYL : -6.00 to +6.00
    sph_cyl_values = [round(x * 0.25, 2) for x in range(-24, 25)]
    
    # Create a unique set of JOB_IDs to ensure uniqueness
    job_ids = set()
    while len(job_ids) < num_records:
        job_ids.add(fake.unique.random_int(100000, 999999))
    job_ids = list(job_ids)

    # Generate records
    for i in range(num_records):
        record = {
            "JOB_ID": job_ids[i],
            "TRAY_ID": fake.random_int(1000, 9999),
            "PATIENT_FIRST_NAME": fake.first_name(),
            "DIA": random.choice(dia_choices),
            "MANF": random.choice(manf_choices),
            "OD_SPH": random.choice(sph_cyl_values),
            "OS_SPH": random.choice(sph_cyl_values),
            "OD_CYL": random.choice(sph_cyl_values),
            "OS_CYL": random.choice(sph_cyl_values),
            "MATERIAL": random.choice(material_choices),
            "OPERATOR": random.choice(operator_choices),
            "QUALITY": random.choice(quality_choices),
            
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    df.to_csv(FILE_PATH, index=False)
    # return df

def collect_data():
    df = pd.read_csv(FILE_PATH)
    return df

if __name__ == '__main__':
    
    # generate_and_save_data(num_records=100) #uncomment when data needs to be generated
    
    df_records = collect_data()
    print(df_records.head())
