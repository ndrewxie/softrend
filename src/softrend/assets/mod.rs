use lazy_static::lazy_static;
use std::collections::HashMap;
use std::fmt;

pub const TEX_SIZE: usize = 128;
#[derive(Clone)]
pub struct Texture(pub Vec<u8>);

impl fmt::Debug for Texture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tex")
    }
}

lazy_static! {
    pub static ref TEXTURES: HashMap<&'static str, Texture> = {
        let mut m = HashMap::new();
        m.insert("joemama", Texture(include_bytes!("./joemama.rgba").to_vec()));
        m.insert(
            "checkerboard",
            Texture(include_bytes!("./checkerboard.rgba").to_vec()),
        );
        m.insert("amogus", Texture(include_bytes!("./amogus.rgba").to_vec()));
        m.insert("white", Texture(vec![255; 128 * 128 * 4]));
        m
    };
}
