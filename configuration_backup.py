"""Versioned configuration backup with validation before an explicit restore."""
import copy
import json
import time
from flask import Blueprint, Response, jsonify, request
from training_service import atomic_json, bounded


def register_backup(app, context):
    bp = Blueprint("configuration_backup", __name__, url_prefix="/api/backup")
    service = context["training_service"]

    def validate():
        if request.content_length is None or request.content_length > 5 * 1024**2:
            raise ValueError("Configuration backups must be smaller than 5 MB")
        payload = request.get_json()
        if not isinstance(payload, dict):
            raise ValueError("Expected a configuration backup")
        backup = payload.get("backup")
        if not isinstance(backup, dict) or backup.get("format") != "ai-camera-configuration" or backup.get("version") != 1:
            raise ValueError("Unsupported configuration backup format or version")
        original = backup.get("settings")
        if not isinstance(original, dict) or any(key not in original for key in context["DEFAULTS"]):
            raise ValueError("Backup settings are incomplete")
        if not isinstance(original["cameras"], list) or len(original["cameras"]) > 100:
            raise ValueError("Backup may contain at most 100 cameras")
        settings = context["clean_settings"](original)
        if len(settings["cameras"]) != len(original["cameras"]) or any(not c.get("id") for c in original["cameras"]):
            raise ValueError("Every restored camera needs an ID and source")
        # Blank secrets intentionally remain blank in a backup, instead of keeping unrelated local tokens.
        for key in context["SECRETS"]:
            if not isinstance(original[key], str):
                raise ValueError("Invalid backup credential")
            settings[key] = original[key]
        if not settings["username"] or not settings["password"]:
            raise ValueError("Backup login must include a username and password")
        rules = backup.get("auto_collect", {})
        if not isinstance(rules, dict):
            raise ValueError("Invalid training collection configuration")
        ids = {c["id"] for c in settings["cameras"]}
        cleaned = {}
        for cid, rule in rules.items():
            if cid not in ids or not isinstance(rule, dict) or rule.get("split") not in {"train", "val", "test"}:
                raise ValueError("Training collection rules must reference restored cameras and valid splits")
            cleaned[cid] = {"enabled": bool(rule.get("enabled")), "split": rule["split"],
                            "interval": bounded(rule.get("interval"), 10, 3600),
                            "confidence": bounded(rule.get("confidence"), .05, .95, float)}
        return payload, settings, cleaned

    @bp.errorhandler(ValueError)
    @bp.errorhandler(TypeError)
    @bp.errorhandler(KeyError)
    @bp.errorhandler(AttributeError)
    def invalid(exc):
        return jsonify(error="Invalid backup: " + str(exc)), 400

    @bp.get("")
    def download():
        with service.lock:
            ids = {c["id"] for c in context["settings"]["cameras"]}
            data = {"format": "ai-camera-configuration", "version": 1, "created": time.time(),
                    "settings": copy.deepcopy(context["settings"]),
                    "auto_collect": {cid: copy.deepcopy(rule) for cid, rule in service.state["auto_collect"].items() if cid in ids}}
        return Response(json.dumps(data, indent=2, allow_nan=False), mimetype="application/json", headers={
            "Content-Disposition": 'attachment; filename="ai-camera-configuration.json"', "Cache-Control": "no-store"})

    @bp.post("/preview")
    def preview():
        _, settings, rules = validate()
        return jsonify(cameras=len(settings["cameras"]), collection_rules=len(rules),
                       message="Replaces saved settings, login, camera credentials, zones, schedules and collection rules. "
                               "Recordings, datasets, model files and the Roboflow connection are not restored.")

    @bp.post("/restore")
    def restore():
        payload, settings, rules = validate()
        if payload.get("confirm") is not True:
            return jsonify(error="Preview the backup and confirm replacement first"), 400
        with service.lock, context["custom_lock"]:
            if any(j["status"] in {"queued", "running", "cancelling"} for j in service.state["jobs"].values()):
                return jsonify(error="Wait for training or transfer jobs to finish before restoring"), 409
            previous = copy.deepcopy(service.state)
            updated = copy.deepcopy(previous)
            updated["auto_collect"] = rules
            updated["history"] = []  # rollback references the configuration being replaced
            try:
                atomic_json(service.state_file, updated)
                try:
                    context["save_settings"](settings)
                except Exception:
                    atomic_json(service.state_file, previous)
                    raise
            except OSError:
                return jsonify(error="Could not restore configuration. Check storage space and permissions"), 503
            service.state = updated
            context["settings"] = settings
            context["custom_det"], context["custom_error"] = None, ""
            context["last_alert"].clear()
            service.last_sample.clear()
            service.last_signature.clear()
        context["sync_cameras"]()
        return jsonify(ok=True, message="Configuration restored. Log in with the restored credentials. Restart the app to reload base AI model paths.")

    app.register_blueprint(bp)
