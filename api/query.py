from lanka_data import CommandRunner


def handler(request):
    path = request.path.replace("/api/", "")

    try:
        result = CommandRunner.run(path)
        return result, 200
    except Exception as e:
        return {"error": str(e)}, 400
