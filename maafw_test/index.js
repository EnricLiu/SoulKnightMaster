import * as maa from '@maaxyz/maa-node'
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log(maa.Global.version)

async function main() {
    // 查询所有Adb设备
    const devices = await maa.AdbController.find()
    if (!devices || devices.length === 0) return;

    // 使用第一个设备创建控制器
    const [name, adb_path, address, screencap_methods, input_methods, config] = devices[0];
    const ctrl = new maa.AdbController(
        adb_path,
        address,
        screencap_methods,
        input_methods,
        config
    );
    let ctrl_ok, res_ok, tskr_ok = false;

    ctrl.notify = (msg, detail) => {
        ctrl_ok = true
        console.log("[INFO] CTRL", msg, detail)
    }
    // 连接设备
    await ctrl.post_connection();
    
    // 创建资源
    const res = new maa.Resource()
    res.notify = (msg, detail) => {
        res_ok = true
        console.log("[INFO] RES ", msg, detail)
    }
    // 加载资源
    await res.post_path(path.resolve(__dirname, 'res'));

    // 创建实例
    const tskr = new maa.Tasker()
    tskr.notify = (msg, detail) => {
        console.log("[INFO] TSKR ", msg, detail)
    }
    
    // 绑定控制器和资源
    tskr.bind(ctrl);
    tskr.bind(res);

    // 执行任务, Task1定义在pipeline/Task.json
    while(true) {
        await new Promise(resolve => setTimeout(resolve, 100));
        if(tskr.inited) break;
    }

    if(await tskr.post_pipeline('Task1').wait().success) {
        console.log('success!')
    }
}

main()
