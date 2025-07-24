# https://learn.udacity.com/nd608?version=2.0.27&partKey=cd13331&lessonKey=900baf17-e90a-4681-a3aa-e4dd9e0d10f7&conceptKey=8a93b943-b604-4155-b0da-b90ccceb4cf3

from openai import OpenAI
client = OpenAI()

response = client.images.edit(
    model='dall-e-2',
    image=open('sunlit_lounge.png', 'rb'),
    mask=open('mask.png', 'rb'),
    prompt="A sunlit indoe lounge area with a pool containing a flamingo",
    n=1,
    size="1024x1024"
)
image_url = response.data[0].url