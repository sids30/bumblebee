import requests
from bs4 import BeautifulSoup

# Define URLs
urls = {
    "Durham College Website": "https://durhamcollege.ca/",
    "Important Dates 2023–2024": "https://durhamcollege.ca/info-for/current-students/important-dates/important-dates-2023-2024",
    "Important Dates 2024–2025": "https://durhamcollege.ca/info-for/current-students/important-dates/important-dates-2024-2025",
    "Enrolment Services": "https://durhamcollege.ca/mydc/enrolment-services",
    "Campus Resources": "https://durhamcollege.ca/academic-faculties/professional-and-part-time-learning/student-information/campus-resources"
}

# Initialize an empty dictionary to store scraped data
scraped_data = {}

# Send HTTP request for each URL and scrape text
for section, url in urls.items():
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find paragraphs, headings, and tables
    text_content = []
    for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table']):
        text_content.append(element.get_text(strip=True))

    # Combine text content into a single string
    scraped_data[section] = '\n'.join(text_content)

# Save scraped data to text file
with open("durham_college_data.txt", "w") as file:
    for section, data in scraped_data.items():
        file.write(f"{section}:\n{data}\n\n")

print("Save to 'durham_college_data.txt'")
