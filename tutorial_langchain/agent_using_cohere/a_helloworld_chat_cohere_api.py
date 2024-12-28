import cohere 
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# print(f"COHERE_API_KEY:{COHERE_API_KEY}")
co = cohere.Client(
  api_key=COHERE_API_KEY, # This is your trial API key
) 

stream = co.chat_stream( 
  model='command-r-08-2024',
  message='<YOUR MESSAGE HERE>',
  temperature=0.3,
  chat_history=[{"role": "User", "message": "Can you give me a global market overview of the solar panels in 2 lines?"}, {"role": "Chatbot", "message": "Certainly! The solar panel market has been experiencing significant growth and development globally, and it plays a crucial role in the transition towards renewable energy sources. Here\'s an overview:\n\nMarket Size and Growth:\n   - The global solar panel market has witnessed remarkable expansion over the past decade. According to various reports, the market size was valued at several billion dollars in recent years, and it is projected to continue growing at a rapid pace.\n   - The increasing demand for clean and sustainable energy, coupled with government initiatives and incentives, has driven the growth of the solar panel industry.\n\nRegional Analysis:\n   - Asia-Pacific: This region dominates the solar panel market, with countries like China, India, and Japan leading the way. China, in particular, has been a major manufacturer and installer of solar panels, contributing significantly to the global supply.\n   - Europe: Europe has been a pioneer in adopting solar energy, with countries like Germany, Italy, and Spain having a strong presence in the market. The region\'s focus on renewable energy targets has driven the demand for solar panels.\n   - North America: The United States and Canada have seen substantial growth in the solar panel market. Favorable policies, tax incentives, and an increasing awareness of environmental sustainability have boosted the adoption of solar energy.\n   - Middle East and Africa: While the region has vast solar potential, the market is still developing. Countries like the United Arab Emirates and South Africa are making strides in solar energy deployment.\n   - Latin America: Brazil and Mexico are key players in the Latin American solar panel market, with increasing investments and projects.\n\nTechnology and Innovation:\n   - Solar Panel Efficiency: Technological advancements have led to higher efficiency rates in solar panels. Manufacturers are continuously improving cell structures and materials to maximize energy conversion.\n   - Bifacial Solar Panels: These panels can capture sunlight from both sides, increasing energy generation. They are gaining popularity and are expected to contribute to overall market growth.\n   - Perovskite Solar Cells: Researchers are exploring perovskite-based solar cells, which offer potential cost-effectiveness and flexibility. This technology could revolutionize the industry.\n\nMarket Drivers and Challenges:\n   - Environmental Concerns: Growing awareness of climate change and the need to reduce carbon emissions have propelled the demand for solar energy.\n   - Government Support: Many governments offer incentives, subsidies, and feed-in tariffs to encourage the adoption of solar panels, making them more economically viable.\n   - Grid Parity: In some regions, solar energy has achieved grid parity, meaning it is cost-competitive with traditional fossil fuel-based electricity generation.\n   - Supply Chain Issues: The solar panel industry relies on a complex supply chain, and disruptions can impact production and prices.\n   - Competition: The market is highly competitive, with numerous manufacturers, especially in Asia. This competition drives innovation but can also lead to price fluctuations.\n\nFuture Outlook:\n   - The solar panel market is expected to maintain its growth trajectory, with an increasing number of countries setting ambitious renewable energy targets.\n   - The integration of solar panels with other technologies, such as energy storage systems, is likely to gain momentum, ensuring a more stable and reliable energy supply.\n   - Advancements in solar panel manufacturing processes and materials are expected to further enhance efficiency and reduce costs.\n\nIt\'s important to note that the solar panel market is dynamic and influenced by various factors, including policy changes, technological breakthroughs, and global energy trends. Staying updated with the latest developments is crucial for a comprehensive understanding of this rapidly evolving industry."}],
  prompt_truncation='AUTO',
  connectors=[{"id":"web-search"}]
) 

for event in stream:
  if event.event_type == "text-generation":
    print(event.text, end='')