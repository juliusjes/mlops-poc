import asyncio
import logging
import time
import traceback

from ml_platform.inference_server.collector import Collector
from ml_platform.inference_server.config import (
    CollectorConfig,
    DeploymentState,
    EngineConfig,
    PluginConfig,
)
from ml_platform.inference_server.engine import Engine
from ml_platform.inference_server.deployment_lifecycle import cancel_worker_task
from ml_platform.inference_server.observation import Observation
from ml_platform.inference_server.plugins.plugin_logger import PluginLogger
from ml_platform.shared.utils import make_hash

logger = logging.getLogger(__name__)

from ml_platform.inference_server.plugins.drift_detection.drift_detection import (
    DriftDetector,
)
from ml_platform.inference_server.plugins.evaluation.classification_metrics import (
    ClassificationMetrics,
)

plugin_registry = {
    "drift": DriftDetector,
    "eval": ClassificationMetrics
}


class Deployment:
    def __init__(self, uuid, engine_cfg, collector_cfg, plugins_cfgs):
        self.uuid = uuid
        self.state = None
        self._set_state(DeploymentState.DEPLOYING)
        self.engine: Engine = self.init_engine(engine_cfg)

        self.collector: Collector = self.init_collector(collector_cfg)

        self.plugin_logger = PluginLogger()
        self.plugin_logger.init_log()

        self.plugins = None
        self.init_plugins(plugins_cfgs)

        # start observation worker
        self.observation_queue = asyncio.Queue()
        self.worker_task = None

    async def start(self):
        self.worker_task = asyncio.create_task(self.observation_worker())
        self._set_state(DeploymentState.READY)


    async def stop(self):
        """Stop the observation worker owned by this deployment."""
        self.worker_task = await cancel_worker_task(self.worker_task)


    async def observation_worker(self):
        while True:
            obs = await self.observation_queue.get()

            self.collector.collect(obs)

            t0 = time.perf_counter()

            outputs = await asyncio.gather(
                *[
                    asyncio.to_thread(plugin.process)
                    for plugin in self.plugins.values()
                ],
                return_exceptions=True,
            )

            print( "all plugins:", (time.perf_counter() - t0) * 1000, "ms" )

            results = []

            for plugin, output in zip(self.plugins.values(), outputs):
                if isinstance(output, Exception):
                    trace = "".join(
                        traceback.format_exception(
                            type(output),
                            output,
                            output.__traceback__,
                        )
                    )
                    logger.debug(f"Plugin {plugin} failed: {trace}")
                    continue

                if output is None:
                    continue

                results.append(output)
                logger.debug(f"{plugin} ok: {output}")


            self.plugin_logger.collect_results(results)

            self.observation_queue.task_done()


    def _set_state(self, state:DeploymentState):
        if self.state is not state:
            logger.info(f"{self.state or 'None'} -> {state}")
            self.state = state


    def reset_plugin_logger(self) -> tuple[str, str]:
        old_name = self.plugin_logger.flush_and_init()
        return str(self.plugin_logger.log_file), str(old_name)


    def init_engine(self, cfg: EngineConfig):
        try:
            engine = Engine(cfg, self.uuid)
        except Exception:
            self._set_state(DeploymentState.FAILED)
            return None
        logger.info(f"Initialized Engine for {self.uuid}")
        return engine
        

    def init_collector(self, cfg: CollectorConfig):
        try:
            collector = Collector(cfg)
        except Exception:
            self._set_state(DeploymentState.FAILED)
            return None
        logger.info(f"Initialized Collector for {self.uuid}")
        return collector


    def init_plugins(self, cfgs: list[PluginConfig]):
        self.plugins = {}
        for cfg in cfgs:
            success = self.add_plugin(cfg)
            if not success:
                self._set_state(DeploymentState.FAILED)
                return
            
        logger.info(f"Initialized Plugins for {self.uuid}")


    def get_plugins(self):
        # TODO: add config
        return list(self.plugins.keys())


    def add_plugin(self, cfg: PluginConfig):
        # append plugin to self.plugins

        if hasattr(self, "plugins"):
            if cfg.name in self.plugins.keys():
                logger.error(f"Plugin with name {cfg.name} already exists")
                return False

        try:
            cls = plugin_registry.get(cfg.plugin_type)

            if cls is None:
                logger.error(f"plugin_type '{cfg.plugin_type}' does not correspond to registered plugins")
                raise KeyError
            
            plugin = cls(cfg, self.collector)
            cfg.name = cfg.name + str(make_hash(cfg.parameters)[:4])
            self.plugins[cfg.name] = plugin

        except Exception as e:
            logger.error(f"Failed to add plugin to plugins in deployment {str(self.uuid)} {str(e)}")    
            return False
        
        logger.info(f"Added plugin {cfg.name} to plugins in deployment {str(self.uuid)}")
        return True


    def remove_plugin(self, method, name):
        # remove plugin from self.plugins,
        n_before = len(self.plugins.keys())
        to_remove = []
        if method == "all":
            for k in self.plugins:
                to_remove.append(k)

        elif method == "byname":
            for k in self.plugins:
                if k == name:
                    to_remove.append(k)

        elif method == "regex":
            # TODO: build a regex pattern from name and find all matches in self.plugins.keys()
            pass

        for name in to_remove:
            self.plugins.pop(name)
            logger.info(f"Removed {name}")
            
        n_after = len(self.plugins.keys())
        remaining = list(self.plugins.keys())

        return n_before - n_after, remaining


    def add_plugins(self, cfgs):
        # add multiple plugins to self.plugins
        results = []
        for c in cfgs:
            success = self.add_plugin(c)
            results.append(success)
        return results
        

    def get_model_name(self):
        if self.engine is None:
            return None
        return self.engine.get_model_name()


    async def predict(self, x, label=None):
        """
        Predict a sample using an engine. Add the observation to the queue
        """
        if self.state == DeploymentState.READY:
            pred = self.engine._predict(x)

            obs = Observation(
                input=x,
                prediction=pred,
                label=label
            )
            await self.observation_queue.put(obs)
            #logger.debug(f"Added observation to queue")
            return pred
        
        else: 
            return []

    async def sync_observation_queue(self):
        await self.observation_queue.join()

    def reset_collector(self):
        self.collector.reset()

    def reset_plugins(self):
        plugin_configs = [p.config for p in self.plugins.values()]
        _, r = self.remove_plugin(method="all", name="placeholder")
        assert len(r) == 0
        self.add_plugins(plugin_configs)
