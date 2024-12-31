e:
	bpython src/example.py -i

de:
	scp -r src textminr:~/curl

fastapi:
	python -m fastapi dev src/fastapi/main.py

flask:
	python -m flask --app src/flask/main.py run
