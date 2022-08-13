import requests
from requests.exceptions import Timeout
import queue, time, urllib.request
from threading import Thread, Event
from datetime import datetime

def perform_web_requests(addresses, no_workers):
    class Worker(Thread):
        def __init__(self, request_queue):
            Thread.__init__(self)
            self.queue = request_queue
            self.results = []

        def get_url(url):
            return requests.get(url)


        def run(self):
            while True:
                content = self.queue.get()
                print ("content: " + content + "\n")
                if content == "":
                    break
                response = requests.get(content)
                # request = urllib.request.Request(content)
                # response = urllib.request.urlopen(request)
                self.results.append("url: " + content + ", response: " + response.text[0:10])
                self.queue.task_done()

    # Create queue and add addresses
    q = queue.Queue()
    for url in addresses:
        q.put(url)

    # Workers keep working till they receive an empty string
    for _ in range(no_workers):
        q.put("")

    # Create workers and add tot the queue
    workers = []
    for _ in range(no_workers):
        worker = Worker(q)
        worker.start()
        workers.append(worker)
    # Join workers to wait till they finished
    for worker in workers:
        worker.join()

    # Combine results from all workers
    r = []
    for worker in workers:
        r.extend(worker.results)
    return r



urls = [ "https://www.google.com", "https://in.search.yahoo.com" ]
# results = perform_web_requests(urls, 1)
# print (type(results[0]))
# print (results)




# print (type(get_url("https://www.google.com")))
# print (type(get_url("https://rajattjainn.github.io/").text))


def hit_api(url):
    current_time = datetime.now().time()
    # current_time = now.strftime("%H:%M:%S")
    print ("url: " + url + ", time: " + str(current_time))
    response = requests.get(url)
    text = "url: " + url + ", response: " + response.text[0:10]
    print (text)
    return text

# thread1 = Thread(target = hit_api, args = (urls[0], ))
# thread2 = Thread(target = hit_api, args = (urls[1], ))

# current_time = datetime.now().time()
# print ("thread 1 starting, " + str(current_time))
# thread1.start()
# current_time = datetime.now().time()
# print ("thread 1 started, " + str(current_time))

# current_time = datetime.now().time()
# print ("thread 2 starting, " + str(current_time))
# thread2.start()
# current_time = datetime.now().time()
# print ("thread 2 started, " + str(current_time))

# thread1.join()
# thread2.join()

REQUEST_TIMEOUT = 1/20
THREAD_TIMEOUT = 1/10

class RequestThread(Thread):
    def __init__(self, url, iteration) -> None:
        super().__init__()
        self.url = url
        self.iteration = iteration

    def run(self):
        str_time = datetime.now()
        print ("url: " + self.url + ", iteration: " + str(self.iteration) + ", time: " + str(str_time.time()))
        try:
            response = requests.get(self.url, timeout=REQUEST_TIMEOUT)
            end_time = datetime.now()
            text = "got response for url: " + self.url + ", iteration: " + str(self.iteration) + ", time: " + str(end_time.time()) + ", time diff: " + str(end_time - str_time)
            print (text)
            self.response = text
        except Timeout:
            self.response = False
            print ("timed out, url: " + self.url + ", iteration: " + str(self.iteration) + ", time: " + str(str_time.time()))
        

for i in range(3):
    print ("\n\nstartingiteration: " + str(i))

    start_time = datetime.now()

    thread1 = RequestThread(urls[0], i)
    current_time = datetime.now().time()
    print ("thread 1 starting, " + str(current_time))
    thread1.start()
    current_time = datetime.now().time()
    print ("thread 1 started, " + str(current_time))
    thread2 = RequestThread(urls[1], i)

    current_time = datetime.now().time()
    print ("thread 2 starting, " + str(current_time))
    thread2.start()
    current_time = datetime.now().time()
    print ("thread 2 started, " + str(current_time))

    thread1.join(THREAD_TIMEOUT)
    print ("joined thread 1, " + str(current_time))
    thread2.join(THREAD_TIMEOUT)
    print ("joined thread 2, " + str(current_time))

    if (thread1.is_alive() or thread2.is_alive()):
        print ("Threads not returned in given timeframe, exiting this iteration")
        # TODo: kill the threads
    else:
        resp1 = thread1.response
        resp2 = thread2.response

        if (resp1 and resp2):
            print ("Both threads ended, results:")
            print (resp1)
            print (resp2)
        else:
            print ("requests didn't complete in the given time")

        end_time = datetime.now()
        print ("Iteration Time:")
        print (end_time - start_time)

    print ("ending iteration: " + str(i))
